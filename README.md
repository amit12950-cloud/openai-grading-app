# OpenAI Grading App

This repository contains a Flask‑based web application that grades student handwritten solutions to math problems using OpenAI’s generative models and vector embeddings.  Students sign in with their MIT email address and a unique access code, upload pictures of their work, and the application automatically evaluates the submission against a rubric, calculates scores and provides detailed feedback.

## How it works

### User authentication and uploading solutions
* **Access codes and MIT email** – Students must log in with an `@mit.edu` address and a matching access code stored in `userdatabase.csv`.  Flask‑Login manages the session, and the app protects grading routes so unauthenticated users cannot upload or view results.
* **Upload multiple images** – A student may upload one or more images of their handwritten solution.  The server temporarily stores these images and sends them to an external conversion model to extract the raw mathematical text and LaTeX.  The conversion step is invoked via a *conversion prompt* (see the `conversion_prompt` in `flask_app.py`).  An external OCR/LaTeX service (such as GPT‑4 with vision or other OCR tools) must be supplied separately; the repository does not include image‑to‑LaTeX conversion.

### Evaluating against a rubric
* **Generative evaluation** – For each problem there is an official rubric stored in `solution_with_rubric_1.txt`.  The application constructs prompts to instruct GPT‑4 to grade the student’s solution: one prompt summarises the student’s answer and another rates each rubric item as *Excellent*, *Partial* or *Unsatisfactory*.  To improve reliability, the model is run several times (`EVALUATION_RUNS`) and results are averaged using a thread pool.
* **Caching and fuzzy matching** – Before calling the model, the app checks a cache of previous evaluations.  Responses are normalised and compared using RapidFuzz to find similar answers; if a close match is found the cached grade is reused.  This reduces cost and ensures consistent grading for similar answers.
* **Embedding‑based override** – After grading, the code computes embeddings using OpenAI’s `text‑embedding‑ada‑002` model to measure cosine similarity between the student’s solution and the official solution.  Low‑confidence AI scores are overridden by similarity‑based heuristics.
* **Equation skeleton scoring** – To discourage superficial copying, the code extracts LaTeX equations and compares their “skeletons” using the Hungarian algorithm.  A skeleton similarity score is computed.  When the generative model fails, a fallback TF‑IDF cosine similarity of the full texts is used.  The final grade combines AI evaluation, skeleton similarity and embedding‑based overrides.

### Results and feedback
* **Detailed result page** – After grading, the application displays a result page with the combined score, per‑rubric ratings and a performance message.  Students can also download a detailed evaluation report as a text file.
* **Comments and admin controls** – Students may leave anonymous comments after receiving their grade.  Administrators (identified by a password in the `ADMIN_PASSWORD` environment variable) can view grade summaries, reset attempts and upload new rubrics via dedicated routes.  All grades are saved to `grades.csv` for future reference.

## Installation

1. **Clone this repository** and navigate into it:

   ```bash
   git clone https://github.com/amit12950-cloud/openai-grading-app.git
   cd openai-grading-app
   ```

2. **Create a virtual environment** (recommended) and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies** from the provided requirements file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**.  The app relies on several environment variables:

   * `OPENAI_API_KEY` – your OpenAI API key.  Required to call GPT‑4 and the embedding model.
   * `APP_SECRET_KEY` – secret key for Flask sessions.  Generate a random string for production.
   * `ADMIN_PASSWORD` – password used for admin routes to reset attempts or upload new rubrics (optional).

   Export these in your shell before running the app.  For example:

   ```bash
   export OPENAI_API_KEY=<your-openai-key>
   export APP_SECRET_KEY="<generate-a-random-secret>"
   export ADMIN_PASSWORD="<admin-password>"
   ```

5. **Run the application**:

   ```bash
   python flask_app.py
   ```

   The Flask development server listens on port 5000.  Open `http://localhost:5000` in your browser.  Log in with one of the email/code pairs from `userdatabase.csv`.

## Repository structure

- `flask_app.py` – main Flask server for the grading app.  Handles authentication, file uploads, conversion prompt, AI evaluation, caching, similarity scoring and result rendering.
- `requirements.txt` – lists Python dependencies needed to run the application.
- `grades.csv` – CSV file used to store grading records; initially empty.
- `solution_with_rubric_1.txt` – official rubric used to evaluate the first problem (you can add more rubric files following the same naming convention).
- `userdatabase.csv` – contains user email addresses and corresponding access codes.
- `templates/` – HTML templates for the user interface (login page, upload form, result view, admin panels, etc.).

## Notes

- For a production deployment, configure HTTPS and set a strong `APP_SECRET_KEY`.  Consider running the app behind a WSGI server such as Gunicorn.

