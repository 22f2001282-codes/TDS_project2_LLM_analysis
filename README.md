TDS – LLM Analysis Quiz Solver
by Shristi Patel

A FastAPI-based automated solver for the Tools in Data Science (TDS) – LLM Analysis Quiz, using LLM reasoning + Python tools to solve multi-step data questions.

## 📌 Overview

This project implements an automated system that can solve the TDS LLM Analysis Quiz, which involves steps like:

Reading quiz pages

Understanding tasks

Downloading files (PDF, CSV, audio)

Performing data analysis

Submitting results

Continuing to the next questions until completion

The quiz requires using LLMs for analysis & interpretation, and Python tools for accurate data extraction and computation.

This solution uses a hybrid design:

🔹 LLM (GPT-4o-mini)

For:
✔ Understanding the quiz question
✔ Classifying task type
✔ Explaining what operation should be done

🔹 Python Tools

For:
✔ Scraping HTML
✔ Reading PDFs
✔ Parsing tables
✔ Transcribing audio (GPT-4o-transcribe)
✔ Submitting answers
✔ Handling multi-step quiz chains

This hybrid approach avoids LLM hallucinations and gives accurate, deterministic results.