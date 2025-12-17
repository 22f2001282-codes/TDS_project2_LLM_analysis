🌟 LLM Analysis Quiz Solver — IITM BS (Project 2)

This project is a complete automated quiz-solving system built for the
TDS – LLM Analysis Quiz (Project 2).
It uses Google Gemini, Python tools, PDF/CSV/audio processing, and a FastAPI backend
to solve multi-step quiz pages automatically.

🚀 Project Overview

The quiz provided by IITM contains multiple pages.
Each page may include:

A textual question

A PDF file (tables, numeric data)

A CSV file

An audio file

HTML tables

A submission link that returns the next quiz URL

This project builds a solver that can:

Fetch a quiz page

Read the question

Let an LLM (Gemini) analyze what kind of question it is

Download and process files

Compute the correct answer using Python

Submit the answer

Move to the next page

Continue until the quiz ends

Everything happens automatically.

🧠 What LLM Does in This Project

We use Google Gemini 2.0 Flash (free tier) for:

✔️ Understanding the question

The LLM reads the question text and classifies the task:

pdf_value_sum

table_sum

audio_transcription

image_analysis

general_reasoning

unknown

✔️ Suggesting operations

Example:
“Sum the ‘value’ column on page 2 of the PDF.”

✔️ Giving reasoning

Short explanation for viva documentation.

✔️ Not used for final answer

Actual computations (PDF sum, CSV read, etc.) are done by Python,
which keeps the system:

Robust

Deterministic

And allowed under project rules

If LLM fails, solver safely uses fallback (answer = 42).

🛠️ Tech Stack
Component	Technology
Backend API	FastAPI
LLM	Google Gemini (gemini-2.0-flash)
HTTP Client	httpx
PDF Processing	pdfplumber
CSV Processing	pandas
HTML Parsing	BeautifulSoup4
Environment Management	python-dotenv
Deployment	Render / Local
📁 Project Structure
LLM_Quiz_Project2/
│── main.py             # FastAPI server
│── solver.py           # Complete quiz solving logic (Gemini version)
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation (this file)
│── .env                # Secret variables (not uploaded to Git)

🔑 Environment Setup

Create .env file:

EMAIL=22f2001282@ds.study.iitm.ac.in
SECRET=Dolphin_2025_shristi
GEMINI_API_KEY=YOUR_GEMINI_KEY_HERE

📦 Installation

Create virtual environment

python -m venv venv


Activate it

venv\Scripts\activate   # Windows


Install dependencies

pip install -r requirements.txt

▶️ Run the Server
uvicorn main:app --reload


Open docs:

👉 http://127.0.0.1:8000/docs

🧪 Testing the Solver

Inside Swagger UI, use the /solve endpoint:

Example:

{
  "email": "22f2001282@ds.study.iitm.ac.in",
  "secret": "Dolphin_2025_shristi",
  "url": "https://tds-llm-analysis.s-anand.net/demo"
}


You will receive a detailed JSON report showing:

Question preview

LLM analysis

PDF/CSV/audio details

Computed answer

Submit result

Next quiz URL

🧵 How the Solver Works (Step-by-Step)

Fetch quiz page

Extract question text

LLM analyzes the question

Detects file links:

PDF

Audio

CSV

Downloads file

If PDF → Extract tables → Sum numeric column

If CSV → Load with pandas → Compute value

If audio → placeholder transcript (can be extended)

Submits answer

Moves to next URL

Continues the chain

🎉 Project Completed

This project demonstrates:

LLM reasoning

Tool-use architecture

Automated multi-step web traversal

Real data extraction

Backend engineering

Robust error-handling

Quiz solving pipeline

Perfect for the IITM evaluation and viva exam.

Author
Shristi Patel
LLM Analysis Quiz Project
IITM BS in Data Science (TDS)

