from flask import Flask, request, render_template, redirect, url_for, flash, Response, session
import openai
import base64
import os
import re
import csv
import datetime
import uuid
import numpy as np
import sys
import glob
import random
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading  # For approximate caching thread safety

# Imports for TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

# Use Flask-Session for server-side sessions
from flask_session import Session

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Configure server-side session storage (using filesystem)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Setup Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = "login"

def load_users(csv_filename):
    """Load users from a CSV file."""
    users = {}
    with open(csv_filename, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normalized = {k.strip().lower(): (v.strip() if v else "") for k, v in row.items() if k}
            email = normalized.get("mit email")
            unique_code = normalized.get("unique code")
            if email and unique_code:
                users[email.lower()] = {"access_code": unique_code}
    return users

CSV_FILENAME = os.path.join(os.path.dirname(__file__), "userdatabase.csv")
users_db = load_users(CSV_FILENAME)

# Global cache for grades.csv records
grades_cache = None
def load_grades_cache():
    global grades_cache
    if os.path.exists("grades.csv"):
        with open("grades.csv", "r", newline="") as f:
            reader = csv.reader(f)
            grades_cache = list(reader)
    else:
        grades_cache = []
def get_grades_cache():
    global grades_cache
    if grades_cache is None:
        load_grades_cache()
    return grades_cache
def update_grades_cache():
    load_grades_cache()

class User(UserMixin):
    def __init__(self, email):
        self.id = email
@login_manager.user_loader
def load_user(user_id):
    if user_id in users_db:
        return User(email=user_id)
    return None

def count_attempts(user_email, problem_number):
    count = 0
    records = get_grades_cache()
    for row in records:
        if row and row[1].strip().lower() == user_email.lower() and int(row[6]) == problem_number:
            count += 1
    return count

def get_total_problems():
    rubric_pattern = os.path.join(os.path.dirname(__file__), "solution_with_rubric_*.txt")
    files = glob.glob(rubric_pattern)
    return len(files)

# --- Global Parameters ---
CORRECTNESS_THRESHOLD = 0.3
UNCERTAINTY_THRESHOLD = 0.3
GRADE_BOOLEAN_ACCEPTANCE = True
EVALUATION_RUNS = 4  # Use 4 parallel runs

def dense_cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def compute_similarity(text1, text2):
    try:
        embed_resp1 = openai.Embedding.create(input=text1, model="text-embedding-ada-002")
        embed_resp2 = openai.Embedding.create(input=text2, model="text-embedding-ada-002")
        vec1 = embed_resp1["data"][0]["embedding"]
        vec2 = embed_resp2["data"][0]["embedding"]
        return dense_cosine_similarity(vec1, vec2)
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return 1.0

def override_score_based_on_similarity(official_text, handwritten_text, evaluation_text):
    similarity_value = compute_similarity(official_text, handwritten_text)
    uncertainty_value = 1 - similarity_value
    if not re.search(r"Score:\s*\d+", evaluation_text):
        return evaluation_text, similarity_value, uncertainty_value
    if (uncertainty_value > UNCERTAINTY_THRESHOLD) or (GRADE_BOOLEAN_ACCEPTANCE and similarity_value < CORRECTNESS_THRESHOLD):
        modified_text = re.sub(r"(Score:\s*)\d+", r"\g<1>0", evaluation_text)
        modified_text += f"\nNote: Score overridden to 0 due to uncertainty ({uncertainty_value:.2f}) or low similarity ({similarity_value:.2f})."
        return modified_text, similarity_value, uncertainty_value
    return evaluation_text, similarity_value, uncertainty_value

def calculate_total_score(evaluation_text):
    cleaned_text = evaluation_text.replace("**", "")
    scores = re.findall(
        r"Score:\s*(?:\\textbf\{)?\s*([\d]+)\s*(?:\})?\s*/\s*(?:\\textbf\{)?\s*([\d]+)\s*(?:\})?",
        cleaned_text,
        flags=re.UNICODE
    )
    earned_total = sum(int(earned) for earned, _ in scores)
    max_total = sum(int(max_points) for _, max_points in scores)
    return earned_total, max_total

def aggregate_evaluation_results(evaluation_texts):
    scores = [calculate_total_score(text)[0] for text in evaluation_texts]
    avg_score = np.mean(scores) if scores else 0
    std_score = np.std(scores) if scores else 0
    return avg_score, std_score

def extract_equations(text):
    equations = re.findall(r'\\\[(.*?)\\\]', text, re.DOTALL)
    return [eq.strip() for eq in equations]

def clean_equation(equation_str):
    eq = equation_str.replace("\n", " ")
    eq = re.sub(r'\\text\{.*?\}', '', eq)
    eq = re.sub(r'\\quad|\\,|\\;|\\!', '', eq)
    return eq.strip()

def tokenize_equation(equation_str):
    tokens = re.findall(r'[A-Za-z0-9_]+|[=()+\-*/^]', equation_str)
    return tokens

def skeleton_similarity(eq1, eq2):
    t1 = tokenize_equation(clean_equation(eq1))
    t2 = tokenize_equation(clean_equation(eq2))
    joined1 = " ".join(t1)
    joined2 = " ".join(t2)
    score = fuzz.ratio(joined1, joined2)
    return score

def token_set_similarity(eq1, eq2):
    tokens1 = set(tokenize_equation(clean_equation(eq1)))
    tokens2 = set(tokenize_equation(clean_equation(eq2)))
    if not tokens1 and not tokens2:
        return 100.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    if not union:
        return 0.0
    return (len(intersection) / len(union)) * 100

def skeleton_based_score(eqs_official, eqs_student):
    if not eqs_official:
        return 0.0, "No equations available for skeleton-based evaluation."
    n = len(eqs_official)
    m = len(eqs_student)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            sim = skeleton_similarity(eqs_official[i], eqs_student[j])
            if sim == 0:
                sim = token_set_similarity(eqs_official[i], eqs_student[j])
            cost_matrix[i, j] = 100 - sim
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    scores = []
    details_list = []
    for i, j in zip(row_ind, col_ind):
        sim = 100 - cost_matrix[i, j]
        scores.append(sim)
        details_list.append(f"Official Equation {i+1} paired with Student Equation {j+1}: Skeleton similarity = {sim:.2f}")
    avg_score = np.mean(scores) if scores else 0.0
    overall_details = "\n".join(details_list)
    return avg_score, overall_details

evaluation_cache = {}
cache_lock = threading.Lock()

def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.lower().split())

def get_approx_cached_evaluation(prompt: str, similarity_threshold: float = 95) -> str:
    normalized_prompt = normalize_prompt(prompt)
    with cache_lock:
        for cached_prompt, response in evaluation_cache.items():
            if fuzz.ratio(normalized_prompt, cached_prompt) >= similarity_threshold:
                return response
    return None

def store_approx_cached_evaluation(prompt: str, response: str):
    normalized_prompt = normalize_prompt(prompt)
    with cache_lock:
        evaluation_cache[normalized_prompt] = response

@app.template_filter('split_str')
def split_str(value, delimiter):
    try:
        return value.split(delimiter)
    except Exception:
        return []

def get_latest_record(user_email, problem_number):
    records = get_grades_cache()
    latest_record = None
    latest_time = None
    for row in records:
        if row and row[1].strip().lower() == user_email.lower() and int(row[6]) == problem_number:
            try:
                t = datetime.datetime.fromisoformat(row[5])
                if latest_time is None or t > latest_time:
                    latest_time = t
                    latest_record = row
            except Exception:
                continue
    return latest_record

@lru_cache(maxsize=None)
def get_rubric(problem_number):
    rubric_file = os.path.join(os.path.dirname(__file__), f"solution_with_rubric_{problem_number}.txt")
    try:
        with open(rubric_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None

def get_device_flags():
    is_mobile = "iphone" in request.user_agent.string.lower()
    accessibility = request.args.get("accessibility", "0") == "1"
    return is_mobile, accessibility

@app.route("/login", methods=["GET", "POST"])
def login():
    is_mobile, accessibility = get_device_flags()
    if request.method == "POST":
        email = request.form.get("email").strip().lower()
        code = request.form.get("code").strip()
        if not email.endswith("@mit.edu"):
            flash("Email must be an MIT email address.")
            return redirect(url_for("login"))
        user_data = users_db.get(email)
        if not user_data:
            flash("Email not found. Please use your MIT email.")
            return redirect(url_for("login"))
        if user_data["access_code"] == code:
            user = User(email=email)
            login_user(user)
            return redirect(url_for("welcome"))
        else:
            flash("Incorrect access code.")
            return redirect(url_for("login"))
    return render_template("login.html", is_mobile=is_mobile, accessibility=accessibility)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# Read the OpenAI API key from an environment variable rather than hard‑coding it.
# When running the application you must set the OPENAI_API_KEY environment
# variable (e.g. ``export OPENAI_API_KEY=<your-key>``) so the OpenAI client
# can authenticate. If the key is not provided the API calls will fail.
openai.api_key = os.environ.get('OPENAI_API_KEY')
client = openai

def encode_image(file_data):
    return base64.b64encode(file_data).decode("utf-8")

@app.route("/")
@login_required
def welcome():
    is_mobile, accessibility = get_device_flags()
    welcome_file = os.path.join(os.path.dirname(__file__), "welcome.txt")
    try:
        with open(welcome_file, "r") as f:
            welcome_message = f.read()
    except FileNotFoundError:
        welcome_message = "Welcome! Please proceed with your submission."
    return render_template("welcome.html", welcome_message=welcome_message, email=current_user.id,
                           is_mobile=is_mobile, accessibility=accessibility)

def process_text(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)

@app.route("/problem/<int:problem_number>", methods=["GET", "POST"])
@login_required
def problem(problem_number):
    is_mobile, accessibility = get_device_flags()
    total_problems = get_total_problems()
    if problem_number < 1 or problem_number > total_problems:
        return "Invalid problem number.", 400

    if request.method == "GET":
        if count_attempts(current_user.id, problem_number) > 0:
            update_grades_cache()
            record = get_latest_record(current_user.id, problem_number)
            if record:
                try:
                    earned_total, max_total = map(int, record[2].split('/'))
                except Exception:
                    earned_total, max_total = 0, 0
                final_score_eval = float(record[11])
                final_score_combined = float(record[12])
                percentage = final_score_eval
                if percentage <= 40:
                    performance_message = "unsatisfactory performance"
                elif percentage <= 80:
                    performance_message = "satisfactory performance"
                else:
                    performance_message = "excellent work"
                return render_template("result.html",
                                       performance_message=performance_message,
                                       detailed_evaluation=record[4],
                                       student_evaluation=record[4],
                                       final_score_skeleton=record[7] if len(record) > 7 else "N/A",
                                       handwritten_explanation="",
                                       handwritten_images=record[3].split("||"),
                                       problem_number=problem_number,
                                       earned_total=earned_total,
                                       max_total=max_total,
                                       total_problems=total_problems,
                                       final_score_eval=final_score_eval,
                                       final_score_combined=final_score_combined,
                                       submission_id=record[0],
                                       is_mobile=is_mobile, accessibility=accessibility)
        else:
            rubric_text = get_rubric(problem_number)
            problem_name = f"Problem {problem_number}"
            if rubric_text:
                for line in rubric_text.splitlines():
                    stripped_line = line.strip()
                    if stripped_line and not all(ch == '-' for ch in stripped_line):
                        problem_name = stripped_line
                        break
            else:
                problem_name = f"Problem {problem_number}"
            return render_template("index.html", problem_number=problem_number, problem_name=problem_name,
                                   attempts_exceeded=False, total_problems=total_problems,
                                   is_mobile=is_mobile, accessibility=accessibility)

    if request.method == "POST":
        if count_attempts(current_user.id, problem_number) > 0:
            flash("You have already submitted an answer for this problem.")
            return redirect(url_for("problem", problem_number=problem_number))
        files = request.files.getlist("handwritten_image")
        if not files or len(files) == 0:
            return "No file part", 400

        base64_images = []
        raw_files = []
        for file in files:
            if file.filename == "":
                continue
            data = file.read()
            raw_files.append((file.filename, data))
            base64_images.append(encode_image(data))
        if not base64_images:
            return "No valid images uploaded.", 400

        submission_id = str(uuid.uuid4())
        # Dropbox upload code removed.

        official_explanation = get_rubric(problem_number)
        if official_explanation is None:
            return f"Rubric file for problem {problem_number} not found.", 500

        conversion_prompt = (
            "Your task is to extract the mathematical content and texts from the attached handwritten solution images. "
            "Convert all equations, expressions, and symbols into LaTeX format, and wrap each one individually using MathJax display math delimiters: \\[ ... \\]. "
            "Return all content, including explanatory text, diagrams (as text description if needed), and derivations. "
            "Ensure all mathematical content and texts are captured in full detail, including steps, derivations, and final results. "
            "Match and format the handwritten solution's math and texts accordingly, without omitting any parts."
        )
        image_messages = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images]
        handwritten_response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            max_tokens=10000,
            messages=[{"role": "user", "content": [{"type": "text", "text": conversion_prompt}] + image_messages}]
        )
        handwritten_explanation_full = handwritten_response.choices[0].message.content
        if '--- Notes ---' in handwritten_explanation_full:
            parts = handwritten_explanation_full.split('--- Notes ---', 1)
            handwritten_explanation = parts[0].strip()
        else:
            handwritten_explanation = handwritten_explanation_full

        session['submission_id'] = submission_id
        session['base64_images'] = base64_images
        session['handwritten_explanation'] = handwritten_explanation
        session['problem_number'] = problem_number

        total_problems = get_total_problems()

        return render_template("conversion_approval.html",
                               conversion_text=handwritten_explanation,
                               base64_images=base64_images,
                               problem_number=problem_number,
                               total_problems=total_problems,
                               is_mobile=get_device_flags()[0],
                               accessibility=get_device_flags()[1])

@app.route("/problem/<int:problem_number>/finalize", methods=["POST"])
@login_required
def finalize_submission(problem_number):
    submission_id = session.get('submission_id')
    base64_images = session.get('base64_images')
    original_conversion = session.get('handwritten_explanation')
    approved_text = request.form.get("approved_text", original_conversion)
    additional_comments = request.form.get("additional_comments", "")
    
    final_student_solution = approved_text
    if additional_comments.strip():
        final_student_solution += "\n\nAdditional Comments:\n" + additional_comments.strip()

    official_explanation = get_rubric(problem_number)
    if official_explanation is None:
        return f"Rubric file for problem {problem_number} not found.", 500

    evaluation_prompt = (
        "You are a grader evaluating a student's solution using the official rubric provided below.\n"
        "For each rubric item, use your reasoning to evaluate the student's solution and choose exactly one of the following options: 'Excellent', 'Partial', or 'Unsatisfactory'.\n"
        "Then, output the following for each rubric item:\n"
        "1. The complete descriptive sentence from the rubric that corresponds to the option you chose ('Excellent', 'Partial', or 'Unsatisfactory').\n"
        "2. A final score line in the exact format: 'Score: [Points Earned]/[Maximum Points]'.\n"
        "3. A detailed explanation outlining the reasoning behind your choice for that rubric item.\n"
        "Do not add any extra commentary beyond the instructions above. Be generous.\n\n"
        "Official Rubric:\n" + official_explanation + "\n\n"
        "Student's Approved Solution:\n" + final_student_solution + "\n\n"
        "Note: Please pay particular attention to the student's additional comments regarding the conversion. If the student noted any misinterpretations of symbols or variables, reflect these concerns appropriately in your evaluation."
    )

    def get_evaluation(prompt):
        cached_response = get_approx_cached_evaluation(prompt)
        if cached_response:
            return cached_response
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[{"role": "user", "content": evaluation_prompt}]
            )
            result = response.choices[0].message.content
            store_approx_cached_evaluation(prompt, result)
            return result
        except Exception as e:
            print(f"Evaluation run failed: {e}")
            return ("Score: 0/0\nFeedback: Evaluation failed due to API error.\n"
                    "Suggestion: Please try resubmitting your solution.\nTOTAL: 0/0")
    
    with ThreadPoolExecutor(max_workers=EVALUATION_RUNS) as executor:
        futures = [executor.submit(get_evaluation, evaluation_prompt) for _ in range(EVALUATION_RUNS)]
        eval_runs = [future.result() for future in as_completed(futures)]
    
    evaluation_output = eval_runs[0]
    handwritten_explanation = final_student_solution
    evaluation_output, similarity_value, uncertainty_value = override_score_based_on_similarity(
        official_explanation, handwritten_explanation, evaluation_output
    )
    avg_earned, score_std = aggregate_evaluation_results(eval_runs)
    
    earned_total, max_total = calculate_total_score(evaluation_output)
    final_score_eval = (earned_total / max_total) * 100 if max_total > 0 else 0

    eqs_official = extract_equations(official_explanation)
    eqs_student = extract_equations(final_student_solution)
    if not eqs_official:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([official_explanation, final_student_solution])
        sim = sklearn_cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        final_score_skeleton = sim * 100
        skeleton_details = f"Non-AI (TF-IDF Cosine Similarity) Score computed on full texts: {final_score_skeleton:.2f}"
    else:
        final_score_skeleton, skeleton_details = skeleton_based_score(eqs_official, eqs_student)

    final_score_combined = (final_score_eval + final_score_skeleton) / 2

    timestamp = datetime.datetime.now().isoformat()
    joined_images = "||".join(base64_images)
    with open("grades.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            submission_id, current_user.id, f"{earned_total}/{max_total}", joined_images,
            evaluation_output, timestamp, problem_number,
            similarity_value, uncertainty_value, avg_earned, score_std,
            final_score_eval, final_score_combined
        ])
    update_grades_cache()

    session.pop('submission_id', None)
    session.pop('base64_images', None)
    session.pop('handwritten_explanation', None)
    session.pop('problem_number', None)

    if final_score_eval <= 40:
        performance_message = "unsatisfactory performance"
    elif final_score_eval <= 80:
        performance_message = "satisfactory performance"
    else:
        performance_message = "excellent work"

    detailed_evaluation = re.sub(r"^TOTAL:\s*\d+\s*/\s*\d+\s*$", "", evaluation_output, flags=re.MULTILINE)
    detailed_evaluation = process_text(detailed_evaluation)
    student_evaluation = re.sub(r"^Score:\s*\d+\s*/\s*\d+\s*$", "", evaluation_output, flags=re.MULTILINE)
    student_evaluation = re.sub(r"^TOTAL:\s*\d+\s*/\s*\d+\s*$", "", student_evaluation, flags=re.MULTILINE)
    student_evaluation = '\n'.join(line for line in student_evaluation.split('\n') if line.strip())
    student_evaluation = process_text(student_evaluation)

    return render_template("result.html",
                           performance_message=performance_message,
                           detailed_evaluation=detailed_evaluation,
                           student_evaluation=student_evaluation,
                           final_score_skeleton=final_score_skeleton,
                           handwritten_explanation=final_student_solution,
                           handwritten_images=base64_images,
                           problem_number=problem_number,
                           earned_total=earned_total,
                           max_total=max_total,
                           total_problems=get_total_problems(),
                           final_score_eval=final_score_eval,
                           avg_earned=avg_earned,
                           score_std=score_std,
                           submission_id=submission_id,
                           is_mobile=get_device_flags()[0],
                           accessibility=get_device_flags()[1])

@app.route("/reset_attempts/<int:problem_number>", methods=["POST"])
@login_required
def reset_attempts(problem_number):
    is_mobile, accessibility = get_device_flags()
    admin_password = request.form.get("admin_password", "")
    if admin_password != "tiwari123":
        flash("Unauthorized: Incorrect admin password.")
        return redirect(url_for("problem", problem_number=problem_number))
    student_email = request.form.get("student_email", "").strip().lower()
    if not student_email:
        flash("Student email is required.")
        return redirect(url_for("admin_controls"))
    try:
        if os.path.exists("grades.csv"):
            with open("grades.csv", "r", newline="") as f:
                reader = csv.reader(f)
                records = list(reader)
            filtered_records = [
                record for record in records
                if not (record[1].strip().lower() == student_email and int(record[6]) == problem_number)
            ]
            with open("grades.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(filtered_records)
        flash(f"Submission attempts for student {student_email} on Problem {problem_number} have been reset.")
    except Exception as e:
        print(f"Error resetting attempts: {e}")
        flash("An error occurred while resetting the attempts.")
    update_grades_cache()
    return redirect(url_for("admin_controls"))

@app.route("/admin", methods=["GET", "POST"])
@login_required
def admin_controls():
    is_mobile, accessibility = get_device_flags()
    if request.method == "POST":
        admin_password = request.form.get("admin_password", "")
        if admin_password != "tiwari123":
            flash("Incorrect admin password.")
            return redirect(url_for("admin_controls"))
        return render_template("admin_controls.html", is_mobile=is_mobile, accessibility=accessibility)
    return render_template("admin_login.html", is_mobile=is_mobile, accessibility=accessibility)

@app.route("/upload_rubrics", methods=["POST"])
@login_required
def upload_rubrics():
    is_mobile, accessibility = get_device_flags()
    admin_password = request.form.get("admin_password", "")
    if admin_password != "tiwari123":
        flash("Unauthorized: Incorrect admin password.")
        return redirect(url_for("admin_controls"))
    files = request.files.getlist("rubric_files")
    if not files:
        flash("No files uploaded.")
        return redirect(url_for("admin_controls"))
    for file in files:
        filename = file.filename
        if not filename.endswith(".txt"):
            continue
        save_path = os.path.join(os.path.dirname(__file__), filename)
        file.save(save_path)
        problem_number = re.findall(r'\d+', filename)
        if problem_number:
            get_rubric.cache_clear()
    flash("Rubrics uploaded successfully.")
    return redirect(url_for("admin_controls"))

@app.route("/thankyou")
def thankyou():
    is_mobile, accessibility = get_device_flags()
    message_file = os.path.join(os.path.dirname(__file__), "message.txt")
    try:
        with open(message_file, "r") as f:
            thank_you_message = f.read()
    except FileNotFoundError:
        thank_you_message = "Thank you for your submission!"
    return render_template("thankyou.html", thank_you_message=thank_you_message,
                           is_mobile=is_mobile, accessibility=accessibility)

@app.route("/save_comment", methods=["POST"])
@login_required
def save_comment():
    is_mobile, accessibility = get_device_flags()
    submission_id = request.form.get("submission_id")
    comment_text = request.form.get("comment")
    if not submission_id or not comment_text:
        flash("Submission ID or comment is missing.")
        return redirect(request.referrer)
    comment_file = "comments.csv"
    timestamp = datetime.datetime.now().isoformat()
    with open(comment_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([submission_id, current_user.id, comment_text, timestamp])
    flash("Comment saved successfully.")
    return redirect(request.referrer)

@app.route("/download_evaluation/<submission_id>")
@login_required
def download_evaluation(submission_id):
    evaluation_record = None
    if os.path.exists("grades.csv"):
        with open("grades.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                    evaluation_record = row
                    break
    if not evaluation_record:
        flash("Evaluation record not found.")
        return redirect(url_for("welcome"))
    details = f"Submission ID: {evaluation_record[0]}\n"
    details += f"Student Email: {evaluation_record[1]}\n"
    details += f"Score: {evaluation_record[2]}\n"
    details += f"Timestamp: {evaluation_record[5]}\n"
    details += f"Problem Number: {evaluation_record[6]}\n\n"
    details += "Detailed Evaluation:\n"
    details += f"{evaluation_record[4]}\n\n"
    details += "Final Scores:\n"
    details += f"AI Evaluation Score: {evaluation_record[11]}\n"
    comment = ""
    if os.path.exists("comments.csv"):
        with open("comments.csv", "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == submission_id and row[1].strip().lower() == current_user.id.lower():
                    comment = row[2]
                    break
    if comment:
        details += f"\nComment: {comment}\n"
    return Response(details, mimetype="text/plain",
                    headers={"Content-Disposition": f"attachment;filename=evaluation_{submission_id}.txt"})

if __name__ == "__main__":
    app.run(debug=True)

