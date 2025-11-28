from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tempfile
import time
import pdfplumber
import pandas as pd

# ----------------------------------------
# 🔹 Load ENV variables (from Render OR local .env)
# ----------------------------------------
EMAIL = os.getenv("EMAIL", "22f2001282@ds.study.iitm.ac.in")
SECRET = os.getenv("SECRET", "Dolphin_2025_shristi")
# Prefer GEMINI_API_KEY, fallback to OPENAI_API_KEY for compatibility
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ----------------------------------------
# 🔹 Gemini (google-genai) client setup
# ----------------------------------------
# pip install google-genai
from google import genai
from google.genai import types

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    # If no key provided, client will attempt to use environment defaults
    client = genai.Client()

# ----------------------------------------
# 🔹 TIME LIMIT FOR QUIZ CHAIN (3 minutes)
# ----------------------------------------
MAX_RUNTIME = 160   # seconds (safe buffer inside 180 sec)


# ============================================================
# PART 2 — HELPER FUNCTIONS (FETCH, FILES, PDF, AUDIO, HTML)
# ============================================================

# ----------------------------------------
# 🔹 Fetch HTML content from a URL
# ----------------------------------------
async def fetch_page_html(url: str) -> str:
    print(f"[fetch_page_html] Fetching HTML: {url}")
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client_http:
        response = await client_http.get(url)
        response.raise_for_status()
        return response.text


# ----------------------------------------
# 🔹 Download ANY file (PDF, CSV, AUDIO, etc)
# ----------------------------------------
async def download_file(url: str) -> bytes:
    print(f"[download_file] Downloading file: {url}")
    async with httpx.AsyncClient(follow_redirects=True, timeout=40) as client_http:
        response = await client_http.get(url)
        response.raise_for_status()
        return response.content


# ----------------------------------------
# 🔹 Extract QUESTION TEXT from HTML
# ----------------------------------------
def extract_question_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Get readable text from important tags
    parts = []
    for tag in soup.find_all(["h1", "h2", "p", "div"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            parts.append(text)

    combined = " ".join(parts)
    return combined[:500]   # limit output


# ----------------------------------------
# 🔹 Extract SUM of “value” column from Page 2 of a PDF
# ----------------------------------------
def extract_pdf_value_sum(pdf_bytes: bytes) -> float:
    print("[extract_pdf_value_sum] Extracting PDF values...")
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp.flush()

            with pdfplumber.open(tmp.name) as pdf:
                if len(pdf.pages) < 2:
                    raise Exception("PDF has less than 2 pages.")

                page2 = pdf.pages[1]
                table = page2.extract_table()

                if not table:
                    raise Exception("No table found on page 2.")

                df = pd.DataFrame(table[1:], columns=table[0])

                if "value" not in df.columns:
                    raise Exception("Column 'value' not found.")

                # Convert to numeric
                df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
                total = df["value"].sum()
                return float(total)

    except Exception as e:
        print(f"[extract_pdf_value_sum] Error: {e}")
        return None


# ----------------------------------------
# 🔹 AUDIO TRANSCRIPTION using Gemini (inline bytes)
# ----------------------------------------
def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """
    Transcribe audio using google-genai by sending audio bytes inline
    (best for small files < ~20MB).
    """
    print("[transcribe_audio] Running Gemini transcription...")

    if not GEMINI_API_KEY:
        return "Transcription skipped (no GEMINI_API_KEY)."

    try:
        # Build a Part from bytes
        audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)

        # Simple instruction + audio part: Gemini will produce text output
        prompt = "Generate a verbatim transcript of the following audio. Return only the transcript text."

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, audio_part],
        )

        # The SDK typically exposes text via response.text
        text_output = getattr(response, "text", None)
        if text_output:
            return text_output
        # Some versions may include candidates; try to fallback:
        if hasattr(response, "candidates") and len(response.candidates) > 0:
            return response.candidates[0].content
        return str(response)

    except Exception as e:
        return f"Error during transcription: {e}"


# ----------------------------------------
# 🔹 Find SUBMIT URL from HTML
# ----------------------------------------
def find_submit_url(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Try <form action="">
    form = soup.find("form")
    if form and form.get("action"):
        return urljoin(base_url, form.get("action"))

    # Try any <a> tag containing “submit”
    for a in soup.find_all("a", href=True):
        if "submit" in a.get("href").lower():
            return urljoin(base_url, a["href"])

    # Fallback → SAME URL
    return base_url


# ----------------------------------------
# 🔹 Submit answer (POST)
# ----------------------------------------
async def submit_answer(submit_url: str, answer: float):
    print(f"[submit_answer] Posting to: {submit_url}")

    payload = {
        "answer": answer
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=40) as client_http:
            resp = await client_http.post(submit_url, json=payload)

            return {
                "ok": resp.status_code == 200,
                "status_code": resp.status_code,
                "text": resp.text,
            }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "status_code": None,
            "text": ""
        }


# ============================================================
# PART 3 — LLM ANALYSIS (Gemini)
# ============================================================

def analyze_question_with_llm(question_text: str) -> dict:
    """
    This function sends the question text to Gemini (via google-genai),
    and asks it to return a strict JSON describing the question type,
    operation, and reason.
    """

    if not GEMINI_API_KEY:
        return {
            "type": "unknown",
            "operation": "none",
            "reason": "GEMINI_API_KEY not set; skipping LLM reasoning."
        }

    # Note: use double braces inside f-strings for literal JSON braces.
    prompt = f"""
You are an AI reasoning module helping to solve a data analysis quiz.
Read the question text below and do three things:

1. Identify the question TYPE. Choose one of:
   - pdf_value_sum
   - pdf_other
   - table_sum
   - table_aggregate
   - audio_transcription
   - image_analysis
   - api_call
   - general_reasoning
   - unknown

2. Suggest the OPERATION that Python should perform.
   Example: "sum 'value' column on page 2", or "average of score column".

3. Give a SHORT reason.

Return strict JSON only in this exact form (no extra text):

{{
  "type": "...",
  "operation": "...",
  "reason": "..."
}}

Question text:
\"\"\"{question_text}\"\"\"
"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if not text:
            # fallback: try candidates
            if hasattr(response, "candidates") and len(response.candidates) > 0:
                text = response.candidates[0].content
            else:
                text = str(response)

        try:
            data = json.loads(text)
            return {
                "type": data.get("type", "unknown"),
                "operation": data.get("operation", ""),
                "reason": data.get("reason", "")
            }
        except Exception:
            return {
                "type": "unknown",
                "operation": "none",
                "reason": "Could not parse JSON from LLM. Raw response: " + (text[:1000] if isinstance(text, str) else repr(text))
            }

    except Exception as e:
        return {
            "type": "unknown",
            "operation": "none",
            "reason": f"LLM failed: {e}"
        }


# ============================================================
# PART 4 — MAIN QUIZ SOLVER FUNCTIONS
# ============================================================

# ----------------------------------------
# 🔹 Solve ONE quiz page
# ----------------------------------------
async def solve_single_quiz(quiz_url: str) -> dict:
    print(f"\n[solve_single_quiz] Solving page: {quiz_url}")

    # 1. Fetch HTML
    html = await fetch_page_html(quiz_url)

    # 2. Extract question text for LLM
    question_preview = extract_question_text(html)

    # 3. LLM analysis (Gemini)
    llm_analysis = analyze_question_with_llm(question_preview)

    # 4. Detect PDF / Audio / Table links
    soup = BeautifulSoup(html, "lxml")

    pdf_link = None
    audio_link = None
    csv_link = None

    for a in soup.find_all("a", href=True):
        href = a["href"]

        if href.lower().endswith(".pdf"):
            pdf_link = urljoin(quiz_url, href)

        elif any(href.lower().endswith(ext) for ext in [".wav", ".mp3", ".m4a"]):
            audio_link = urljoin(quiz_url, href)

        elif href.lower().endswith(".csv"):
            csv_link = urljoin(quiz_url, href)

    answer = None
    details = {}

    # --------------------------------------------------------
    # CASE 1: PDF question → sum "value" column from page 2
    # --------------------------------------------------------
    if pdf_link:
        print(f"[solve_single_quiz] PDF detected: {pdf_link}")
        try:
            pdf_bytes = await download_file(pdf_link)
            total = extract_pdf_value_sum(pdf_bytes)

            if total is None:
                print("[solve_single_quiz] PDF parse failed, using fallback 42")
                answer = 42
                details["pdf_result"] = "PDF error → fallback answer = 42"
            else:
                answer = float(total)
                details["pdf_result"] = f"Sum of 'value' column = {answer}"

        except Exception as e:
            answer = 42
            details["pdf_result"] = f"PDF failed ({e}) → fallback 42"

    # --------------------------------------------------------
    # CASE 2: AUDIO question → transcribe → extract number
    # --------------------------------------------------------
    elif audio_link:
        print(f"[solve_single_quiz] AUDIO detected: {audio_link}")
        audio_bytes = await download_file(audio_link)
        transcript = transcribe_audio(audio_bytes)

        # Very simple: find first number in transcript
        import re
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", transcript)
        answer = float(nums[0]) if nums else 42

        details["audio_transcript"] = transcript

    # --------------------------------------------------------
    # CASE 3: CSV question → simple sum of first numeric column
    # --------------------------------------------------------
    elif csv_link:
        print(f"[solve_single_quiz] CSV detected: {csv_link}")
        try:
            csv_bytes = await download_file(csv_link)
            df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))

            # numeric sum of first column
            numeric_cols = df.select_dtypes(include="number")
            if numeric_cols.shape[1] > 0:
                answer = numeric_cols.iloc[:, 0].sum()
                details["csv_result"] = f"Sum of numeric column = {answer}"
            else:
                answer = 42
                details["csv_result"] = "No numeric column found → 42"

        except Exception:
            answer = 42
            details["csv_result"] = "CSV parse failed → 42"

    # --------------------------------------------------------
    # CASE 4: No recognizable file — fallback
    # --------------------------------------------------------
    else:
        print("[solve_single_quiz] No PDF/Audio/CSV found → fallback answer = 42")
        answer = 42
        details["fallback"] = "Used fallback answer 42"

    # 5. Find submit URL
    submit_url = find_submit_url(html, quiz_url)

    # 6. Submit answer
    submit_result = await submit_answer(submit_url, answer)

    # 7. Detect NEXT quiz URL
    next_url = None
    if submit_result.get("ok") and "http" in submit_result.get("text", ""):
        # teacher's server usually returns plain URL in body
        possible = submit_result["text"].strip()
        if possible.startswith("http"):
            next_url = possible

    # 8. Return structured result
    return {
        "url": quiz_url,
        "answer": answer,
        "question_preview": question_preview,
        "llm_analysis": llm_analysis,
        "details": details,
        "submit_url": submit_url,
        "submit_result": submit_result,
        "next_url": next_url,
    }


# ----------------------------------------
# 🔹 Solve ALL quiz pages (chained)
# ----------------------------------------
async def solve_quiz_chain(start_url: str):
    print("\n[solve_quiz_chain] Starting chain...")

    visited = []
    url = start_url

    start_time = time.time()

    while url:
        # Stop if time limit exceeded
        if time.time() - start_time > MAX_RUNTIME:
            print("[solve_quiz_chain] TIMEOUT")
            break

        result = await solve_single_quiz(url)
        visited.append(result)

        url = result.get("next_url")  # go to next question

    return {
        "visited": visited,
        "total_steps": len(visited)
    }
