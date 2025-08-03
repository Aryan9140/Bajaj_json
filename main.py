
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os, requests, fitz, tempfile, asyncio
from dotenv import load_dotenv
from typing import List, Optional, Union
from email import policy
from email.parser import BytesParser
import docx
import urllib.parse
import openai
import re

load_dotenv()

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_TOKEN = os.getenv("API_TOKEN")

openai.api_key = OPENAI_API_KEY

class RunRequest(BaseModel):
    documents: Optional[Union[str, List[str]]] = None
    questions: List[str]
    email_file: Optional[str] = None

PROMPT_TEMPLATE = """
You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

Context:
{context}

Question:
{query}

Instructions:
1. Provide a clear and direct answer based ONLY on the context.
2. Do not specify clause numbers or descriptions.
3. If the answer is "Yes" or "No," include a short explanation based on the clause.
4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
5. Do NOT invent or assume any information outside the given context.
6. Limit each answer to a maximum of one paragraph.
7. If the context is too long, summarize it to focus on relevant parts.

Answer:
"""

def extract_text_from_pdf(pdf_url: str) -> str:
    response = requests.get(pdf_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    text = ""
    doc = fitz.open(tmp_path)
    for page in doc:
        text += page.get_text("text")
    doc.close()
    os.remove(tmp_path)
    return text.strip()

def extract_text_from_docx(docx_url: str) -> str:
    response = requests.get(docx_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    doc = docx.Document(tmp_path)
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    os.remove(tmp_path)
    return text

def extract_text_from_email(email_url: str) -> str:
    response = requests.get(email_url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    with open(tmp_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    os.remove(tmp_path)
    email_text = f"Subject: {msg['subject']}\n\n"
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                email_text += part.get_content()
    else:
        email_text += msg.get_content()
    return email_text.strip()

def get_relevant_context(document_text: str, question: str, max_sents: int = 5) -> str:
    # Split document into sentences (basic splitting)
    sents = re.split(r'(?<=[.!?])\s+', document_text)
    question_keywords = [w for w in re.findall(r'\w+', question.lower()) if len(w) > 2]
    sent_scores = []
    for sent in sents:
        score = sum(1 for w in question_keywords if w in sent.lower())
        sent_scores.append((score, sent))
    sent_scores.sort(reverse=True)  # More relevant first
    top = [s for (score, s) in sent_scores if score > 0][:max_sents]
    if not top:
        top = sents[:max_sents]  # Fallback: first 5 sentences
    return " ".join(top)

async def ask_gpt4o(context, question):
    prompt = PROMPT_TEMPLATE.format(context=context, query=question)
    response = await asyncio.to_thread(
        openai.chat.completions.create,
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

@app.post("/api/v1/hackrx/run")
async def run_analysis(request: RunRequest, authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        combined_text = ""
        doc_urls = []
        if request.documents:
            if isinstance(request.documents, str):
                doc_urls = [request.documents]
            elif isinstance(request.documents, list):
                doc_urls = request.documents
        for doc_url in doc_urls:
            parsed_url = urllib.parse.urlparse(doc_url)
            file_name = os.path.basename(parsed_url.path).lower()
            if file_name.endswith(".pdf"):
                combined_text += "\n" + extract_text_from_pdf(doc_url)
            elif file_name.endswith(".docx"):
                combined_text += "\n" + extract_text_from_docx(doc_url)
            elif file_name.endswith(".txt"):
                combined_text += "\n" + requests.get(doc_url).text
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")
        if request.email_file:
            combined_text += "\n" + extract_text_from_email(request.email_file)
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

        # Extract relevant context per question
        tasks = []
        for q in request.questions:
            context_for_q = get_relevant_context(combined_text, q)
            tasks.append(ask_gpt4o(context_for_q, q))
        answers = await asyncio.gather(*tasks)

        # Join answers as a single string with newlines
        result_string = "\n".join(answers)
        return {"answers": result_string}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
