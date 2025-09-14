# OpenAI Grading App

This repository contains a Flask‑based web application that grades student handwritten solutions to math problems using OpenAI embeddings.  The app authenticates users via MIT email and unique access codes, converts uploaded images to text, compares the student’s solution against an official rubric using vector embeddings, calculates scores, and stores results.

## Features

* **User authentication** – students sign in with their MIT email and a unique access code stored in `userdatabase.csv`.
* **Handwritten solution upload** – users can upload one or more images of their handwritten solution for each problem.
* **Image to LaTeX conversion** – the uploaded images are sent to a conversion prompt that extracts text and equations; this part requires an external model or API capable of OCR and LaTeX generation (not included in this repo).
* **Embedding‑based grading** – the student’s solution is compared against the official rubric using OpenAI `text‑embedding‑ada‑002` embeddings.  Cosine similarity and heuristics are used to compute scores and override low‑confidence answers.
* **Result reporting** – the app displays detailed scoring feedback and stores grades in `grades.csv` for record keeping.

## Installation

1. **Clone the repository** (or download the source code) and navigate into it:

   ```bash
   git clone https://github.com/amit12950-cloud/openai-grading-app.git
   cd openai-grading-app
   ```

2. **Create and activate a Python virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies** using the provided requirements file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key**.  The application reads the OpenAI API key from the `OPENAI_API_KEY` environment variable.  Export your key before running the app:

   ```bash
   export OPENAI_API_KEY=<your-openai-api-key>
   ```

5. **Run the application**:

   ```bash
   python flask_app.py
   ```

   By default the Flask development server will run on port 5000.  Open `http://localhost:5000` in your web browser and log in with one of the email/code pairs from `userdatabase.csv`.

## Repository structure

* `flask_app.py` – main Flask server for the grading app.  This version has been sanitised to remove hard‑coded secrets and reads the OpenAI API key from `OPENAI_API_KEY`.
* `requirements.txt` – lists the Python dependencies required to run the project.
* `grades.csv` – CSV file used to store grading records; initially empty.
* `solution_with_rubric_1.txt` – official rubric used to evaluate the first problem.
* `userdatabase.csv` – contains user email addresses and corresponding access codes.
* `templates/` – HTML templates for the user interface (added in a separate commit).

## Notes

* The OCR/LaTeX conversion step for handwritten images is **not** implemented in this repository.  You will need to integrate an external service or model (such as a hosted GPT‑4 Vision API) to convert images into LaTeX and plain text.
* For production deployment, ensure you set a secure `app.secret_key` in `flask_app.py` and configure HTTPS.  This project is provided for educational purposes.