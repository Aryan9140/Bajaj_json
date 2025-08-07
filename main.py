import os, requests, fitz, tempfile, docx, urllib.parse, io, re, asyncio, httpx
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Union
from email import policy
from email.parser import BytesParser
from rapidfuzz import fuzz
import pytesseract
import random

# For OpenRouter
from openai import OpenAI

load_dotenv()
app = FastAPI()

MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY")
API_TOKEN        = os.getenv("API_TOKEN")

# Set Tesseract PATH for Windows backup
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Tesseract-OCR"

# --- PROMPTING ---
QA_PROMPT = """
You are an expert insurance policy analyst. ONLY answer from the context. Quote all numbers, limits, years, eligibility, conditions, and requirements word-for-word.
If the answer is "Yes" or "No", briefly explain why. If not found, say "Not mentioned in the policy."
Write a detailed but concise paragraph, including all points relevant to the question.

Context:
{context}

Question:
{question}

Answer:
"""

GROQ_RECHECK_PROMPT = """
You are a strict insurance policy answer reviewer. Carefully check the following answer for the user's question, based only on the policy context.
If the answer is not fully correct, not complete, or "Not mentioned in the policy", use ONLY the context below to generate a more complete answer. 
DO NOT invent, add, or summarize anything not in the context.

Context:
{context}

Question:
{question}

Draft Answer:
{draft}

Improved/Corrected Answer:
"""

OPENROUTER_FINAL_PROMPT = """
You are an expert insurance policy chatbot. Given the policy context, the user question, and two draft answers below,
choose or improve the most complete, policy-accurate answer using only the policy context. If "Not mentioned in the policy", recheck with the context.
If both are incomplete, you may rewrite using details ONLY from context.
If no answer can be given, respond with: Not mentioned in the policy.

Policy Context:
{context}

Question:
{question}

Draft Answer 1 (from Mistral or Groq):
{ans1}

Draft Answer 2 (from Groq or Mistral):
{ans2}

Final Best Answer:
"""

class RunRequest(BaseModel):
    documents: Optional[Union[str, List[str]]] = None
    questions: List[str]
    email_file: Optional[str] = None

# ------------- LLM FUNCTIONS ---------------
async def call_mistral(prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-small-latest",
        "temperature": 0.13, "top_p": 1, "max_tokens": 950,
        "messages": [{"role": "user", "content": prompt}]
    }
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

async def call_groq(prompt: str, max_retries=3) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.13, "top_p": 1, "max_tokens": 950,
        "messages": [{"role": "user", "content": prompt}]
    }
    last_error = None
    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code == 429:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"Groq 429 received, retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
                last_error = res
                continue
            try:
                res.raise_for_status()
                return res.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = e
                break
    print("Groq LLM failed after retries:", last_error)
    return "Not mentioned in the policy."

async def call_openrouter_gpt4o(prompt: str) -> str:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )
    completion = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            extra_headers={},
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
    )
    return completion.choices[0].message.content.strip()

# ------------ FILE EXTRACTORS (async for OCR) ----------
async def ocr_page_async(page, dpi=200):
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception as e:
        print(f"OCR error: {e}")
        return ""

async def extract_text_pages_from_pdf(pdf_url: str, ocr_min_len=40, ocr_max_pages=20, dpi=200):
    r = requests.get(pdf_url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    doc = fitz.open(path)
    page_texts, ocr_tasks, ocr_indices = [], [], []
    for i, page in enumerate(doc):
        txt = page.get_text("text") or ""
        if len(txt.strip()) < ocr_min_len and len(ocr_tasks) < ocr_max_pages:
            ocr_tasks.append(ocr_page_async(page, dpi))
            ocr_indices.append(i)
            page_texts.append(None)
        else:
            page_texts.append(txt)
    ocr_results = await asyncio.gather(*ocr_tasks) if ocr_tasks else []
    for idx, text in zip(ocr_indices, ocr_results):
        page_texts[idx] = text
    doc.close()
    os.remove(path)
    return [t or "" for t in page_texts]

def extract_text_from_docx(docx_url: str) -> str:
    r = requests.get(docx_url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    doc = docx.Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    os.remove(path)
    return text

def extract_text_from_txt(txt_url: str) -> str:
    return requests.get(txt_url).text

def extract_text_from_email(email_url: str) -> str:
    r = requests.get(email_url)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as tmp:
        tmp.write(r.content)
        path = tmp.name
    with open(path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    os.remove(path)
    body = f"Subject: {msg['subject']}\n\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body += part.get_content()
    else:
        body += msg.get_content()
    return body.strip()

def get_relevant_context(pages, question, window=3, maxlen=9000):
    best_score, best_idx = -1, 0
    for i, txt in enumerate(pages):
        score = fuzz.token_set_ratio(question, txt)
        if score > best_score:
            best_score = score
            best_idx = i
    start = max(0, best_idx - window)
    end   = min(len(pages), best_idx + window + 1)
    chunk = "\n".join(pages[start:end])
    # Add all lines with numbers, limits, % etc.
    all_lines = [line.strip() for p in pages for line in p.split('\n') if len(line.strip()) > 10]
    pattern = re.compile(r'\d+\s*(day|month|year|%)|sum insured|premium|grace|waiting|limit|sub-limit|renewal|NCD|check[-\s]?up|capped|maximum|minimum|deductible|coverage|room rent|ICU|AYUSH|organ donor|claim|settle|required|document', re.I)
    extra_context = "\n".join(list(dict.fromkeys([line for line in all_lines if pattern.search(line)])))
    return (chunk + "\n" + extra_context)[:maxlen]

@app.get("/")
def health():
    return {"message": "OK"}

@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    doc_urls = [request.documents] if isinstance(request.documents, str) else request.documents
    all_pages, pdf_extract_tasks = [], []
    for doc_url in doc_urls:
        path = urllib.parse.urlparse(doc_url).path.lower()
        if path.endswith(".pdf"):
            pdf_extract_tasks.append(extract_text_pages_from_pdf(doc_url))
        elif path.endswith(".docx"):
            all_pages += [extract_text_from_docx(doc_url)]
        elif path.endswith(".txt"):
            all_pages += [extract_text_from_txt(doc_url)]
        elif path.endswith(".eml"):
            all_pages += [extract_text_from_email(doc_url)]
        else:
            raise HTTPException(400, detail=f"Unsupported format: {doc_url}")
    if pdf_extract_tasks:
        pdf_results = await asyncio.gather(*pdf_extract_tasks)
        for res in pdf_results:
            all_pages += res
    if not all_pages:
        raise HTTPException(status_code=400, detail="No valid content extracted.")

    # -------- Dual LLM, then recheck with OpenRouter GPT-4o --------
    async def get_best_llm_answer(q):
        context = get_relevant_context(all_pages, q, window=3, maxlen=9000)
        mistral_task = asyncio.create_task(call_mistral(QA_PROMPT.format(context=context, question=q)))
        groq_task    = asyncio.create_task(call_groq(QA_PROMPT.format(context=context, question=q)))
        mistral_answer, groq_answer = await asyncio.gather(mistral_task, groq_task)
        # If both say "not mentioned", increase window and retry once
        if ("not mentioned" in mistral_answer.lower() and "not mentioned" in groq_answer.lower()):
            context = get_relevant_context(all_pages, q, window=6, maxlen=15000)
            mistral_answer = await call_mistral(QA_PROMPT.format(context=context, question=q))
            groq_answer    = await call_groq(QA_PROMPT.format(context=context, question=q))
        # Let OpenRouter GPT-4o act as final judge/enhancer
        openrouter_prompt = OPENROUTER_FINAL_PROMPT.format(
            context=context, question=q, ans1=mistral_answer, ans2=groq_answer
        )
        try:
            final_answer = await call_openrouter_gpt4o(openrouter_prompt)
        except Exception as e:
            print("OpenRouter LLM failed:", e)
            # fallback: prefer most detailed non-"not mentioned" answer
            if (mistral_answer and "not mentioned" not in mistral_answer.lower()):
                final_answer = mistral_answer
            elif (groq_answer and "not mentioned" not in groq_answer.lower()):
                final_answer = groq_answer
            else:
                final_answer = "Not mentioned in the policy."
        if not final_answer or len(final_answer) < 10:
            final_answer = mistral_answer or groq_answer or "Not mentioned in the policy."
        return final_answer.strip()

    answers = await asyncio.gather(*[get_best_llm_answer(q) for q in request.questions])
    return {"answers": answers}
