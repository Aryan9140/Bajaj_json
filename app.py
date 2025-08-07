# # # # # # # from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File
# # # # # # # from pydantic import BaseModel
# # # # # # # import os, requests, fitz, tempfile
# # # # # # # from dotenv import load_dotenv
# # # # # # # from typing import List, Optional
# # # # # # # from email import policy
# # # # # # # from email.parser import BytesParser
# # # # # # # import docx
# # # # # # # from typing import List, Optional, Union
# # # # # # # from pydantic import BaseModel
# # # # # # # import urllib.parse

# # # # # # # load_dotenv()

# # # # # # # app = FastAPI()

# # # # # # # MISTRAL_API_KEY = os.getenv("OPENAI_API_KEY")
# # # # # # # API_TOKEN = os.getenv("API_TOKEN")

# # # # # # # # Request schema
# # # # # # # class RunRequest(BaseModel):
# # # # # # #     documents: Optional[Union[str, List[str]]] = None  # URLs for docs
# # # # # # #     questions: List[str]
# # # # # # #     email_file: Optional[str] = None       # Optional email file URL

# # # # # # # # Prompt Template
# # # # # # # PROMPT_TEMPLATE = """
# # # # # # # You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

# # # # # # # Context:
# # # # # # # {context}

# # # # # # # Question:
# # # # # # # {query}

# # # # # # # Instructions:
# # # # # # # 1. Provide a clear and direct answer based ONLY on the context.
# # # # # # # 2. Do not specify clause numbers or descriptions.
# # # # # # # 3. If the answer is "Yes" or "No," include a short explanation based on the clause.
# # # # # # # 4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
# # # # # # # 5. Do NOT invent or assume any information outside the given context.
# # # # # # # 6. Limit each answer to a maximum of one paragraph.
# # # # # # # 7. If the context is too long, summarize it to focus on relevant parts.

# # # # # # # Answer:
# # # # # # # """

# # # # # # # # ---------------- Mistral API ----------------
# # # # # # # # def call_mistral(prompt: str) -> str:
# # # # # # # #     url = "https://api.mistral.ai/v1/chat/completions"
# # # # # # # #     headers = {
# # # # # # # #         "Authorization": f"Bearer {MISTRAL_API_KEY}",
# # # # # # # #         "Content-Type": "application/json"
# # # # # # # #     }
# # # # # # # #     payload = {
# # # # # # # #         "model": "mistral-small-latest",
# # # # # # # #         "temperature": 0.3,
# # # # # # # #         "top_p": 1,
# # # # # # # #         "max_tokens": 500,
# # # # # # # #         "messages": [{"role": "user", "content": prompt}]
# # # # # # # #     }
# # # # # # # #     res = requests.post(url, headers=headers, json=payload)
# # # # # # # #     res.raise_for_status()
# # # # # # # #     return res.json()["choices"][0]["message"]["content"]

# # # # # # # import openai

# # # # # # # def call_openai_gpt4o(prompt: str) -> str:
# # # # # # #     response = openai.chat.completions.create(
# # # # # # #         model="gpt-4o",
# # # # # # #         messages=[
# # # # # # #             {"role": "user", "content": prompt}
# # # # # # #         ],
# # # # # # #         temperature=0.3,
# # # # # # #         max_tokens=500,
# # # # # # #         top_p=1,
# # # # # # #     )
# # # # # # #     return response.choices[0].message.content.strip()


# # # # # # # # ---------------- Document Extractors ----------------
# # # # # # # def extract_text_from_pdf(pdf_url: str) -> str:
# # # # # # #     response = requests.get(pdf_url)
# # # # # # #     response.raise_for_status()
# # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# # # # # # #         tmp.write(response.content)
# # # # # # #         tmp_path = tmp.name

# # # # # # #     text = ""
# # # # # # #     doc = fitz.open(tmp_path)
# # # # # # #     for page in doc:
# # # # # # #         text += page.get_text("text")
# # # # # # #     doc.close()
# # # # # # #     os.remove(tmp_path)
# # # # # # #     return text.strip()

# # # # # # # def extract_text_from_docx(docx_url: str) -> str:
# # # # # # #     response = requests.get(docx_url)
# # # # # # #     response.raise_for_status()
# # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# # # # # # #         tmp.write(response.content)
# # # # # # #         tmp_path = tmp.name

# # # # # # #     doc = docx.Document(tmp_path)
# # # # # # #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# # # # # # #     os.remove(tmp_path)
# # # # # # #     return text

# # # # # # # def extract_text_from_email(email_url: str) -> str:
# # # # # # #     response = requests.get(email_url)
# # # # # # #     response.raise_for_status()
# # # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # # # # # #         tmp.write(response.content)
# # # # # # #         tmp_path = tmp.name

# # # # # # #     with open(tmp_path, 'rb') as f:
# # # # # # #         msg = BytesParser(policy=policy.default).parse(f)
# # # # # # #     os.remove(tmp_path)

# # # # # # #     # Extract email text
# # # # # # #     email_text = f"Subject: {msg['subject']}\n\n"
# # # # # # #     if msg.is_multipart():
# # # # # # #         for part in msg.walk():
# # # # # # #             if part.get_content_type() == 'text/plain':
# # # # # # #                 email_text += part.get_content()
# # # # # # #     else:
# # # # # # #         email_text += msg.get_content()
# # # # # # #     return email_text.strip()

# # # # # # # @app.get("/")
# # # # # # # def read_root():
# # # # # # #     return {"message": "FastAPI is running"}

# # # # # # # # ---------------- API Endpoint ----------------
# # # # # # # @app.post("/api/v1/hackrx/run")
# # # # # # # def run_analysis(request: RunRequest, authorization: str = Header(...)):
# # # # # # #     if authorization != f"Bearer {API_TOKEN}":
# # # # # # #         raise HTTPException(status_code=401, detail="Unauthorized")

# # # # # # #     try:
# # # # # # #         combined_text = ""

# # # # # # #         # Extract from documents
# # # # # # #         doc_urls = []
# # # # # # #         if request.documents:
# # # # # # #             if isinstance(request.documents, str):
# # # # # # #                 doc_urls = [request.documents]
# # # # # # #             elif isinstance(request.documents, list):
# # # # # # #                 doc_urls = request.documents

# # # # # # #         for doc_url in doc_urls:
# # # # # # #             parsed_url = urllib.parse.urlparse(doc_url)
# # # # # # #             file_name = os.path.basename(parsed_url.path).lower()
# # # # # # #             if file_name.endswith(".pdf"):
# # # # # # #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# # # # # # #             elif file_name.endswith(".docx"):
# # # # # # #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# # # # # # #             elif file_name.endswith(".txt"):
# # # # # # #                 combined_text += "\n" + requests.get(doc_url).text
# # # # # # #             else:
# # # # # # #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")

# # # # # # #         # Extract from email if provided
# # # # # # #         if request.email_file:
# # # # # # #             combined_text += "\n" + extract_text_from_email(request.email_file)

# # # # # # #         if not combined_text.strip():
# # # # # # #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

# # # # # # #         context = combined_text.strip()

# # # # # # #         # Format multiple questions as a single multi-question prompt
# # # # # # #         numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])
# # # # # # #         multi_question_prompt = PROMPT_TEMPLATE.format(context=context, query=numbered_questions)
# # # # # # #         multi_answer_response = call_openai_gpt4o(multi_question_prompt)

# # # # # # #         # Try splitting response by question number
# # # # # # #         split_answers = []
# # # # # # #         for i in range(len(request.questions)):
# # # # # # #             prefix = f"{i+1}."
# # # # # # #             next_prefix = f"{i+2}."
# # # # # # #             start = multi_answer_response.find(prefix)
# # # # # # #             end = multi_answer_response.find(next_prefix)
# # # # # # #             if start != -1:
# # # # # # #                 answer = multi_answer_response[start:end].strip()
# # # # # # #                 answer = answer.lstrip(f"{prefix}").strip()
# # # # # # #                 split_answers.append(answer)
        
# # # # # # #         # Fallback if not split properly
# # # # # # #         if not split_answers:
# # # # # # #             split_answers = [multi_answer_response.strip()]

# # # # # # #         return {"answers": split_answers}

# # # # # # #     except Exception as e:
# # # # # # #         raise HTTPException(status_code=500, detail=str(e))















# # # # # # from fastapi import FastAPI, HTTPException, Header
# # # # # # from pydantic import BaseModel
# # # # # # import os, requests, fitz, tempfile
# # # # # # from dotenv import load_dotenv
# # # # # # from typing import List, Optional, Union
# # # # # # from email import policy
# # # # # # from email.parser import BytesParser
# # # # # # import docx
# # # # # # import urllib.parse
# # # # # # import openai

# # # # # # # ========== CONFIG ==========
# # # # # # load_dotenv()
# # # # # # API_TOKEN = os.getenv("API_TOKEN", "YOUR_API_TOKEN")
# # # # # # OPENAI_API_KEYS = [
# # # # # #     os.getenv("OPENAI_API_KEY"),  # Set in .env or environment
# # # # # #     os.getenv("OPENAI_API_KEY"),  # You can add more as backup/test
# # # # # #     # ... add more keys if you want to rotate/test
# # # # # # ]
# # # # # # # ========== HIGHEST PRIORITY MODEL AT THE TOP ==========
# # # # # # OPENAI_MODELS = [
# # # # # #     # "gpt-4-turbo",      # 128k context, best for RAG, paid, fast
# # # # # #     # "gpt-4o",           # New, fast, great reasoning, slightly less context
# # # # # #     # "gpt-4",            # Accurate, expensive, slower, 8k/32k context
# # # # # #     "gpt-3.5-turbo",    # Cheap, very fast, 16k context, lower accuracy
# # # # # # ]

# # # # # # MODEL_NAME = os.getenv("OPENAI_MODEL", OPENAI_MODELS[0])  # Can override with env

# # # # # # # ========== FASTAPI INIT ==========
# # # # # # app = FastAPI()

# # # # # # class RunRequest(BaseModel):
# # # # # #     documents: Optional[Union[str, List[str]]] = None
# # # # # #     questions: List[str]
# # # # # #     email_file: Optional[str] = None

# # # # # # # ========== PROMPT TEMPLATE (RICH, STRICT, INSURANCE) ==========
# # # # # # PROMPT_TEMPLATE = """
# # # # # # You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

# # # # # # Context:
# # # # # # {context}

# # # # # # Question:
# # # # # # {query}

# # # # # # Instructions:
# # # # # # 1. Extract the answer **exactly as written** in the context, with all numbers, durations, percentages, and relevant conditions.
# # # # # # 2. For Yes/No questions, start with "Yes" or "No", then a short explanation.
# # # # # # 3. If the answer is not in the context, reply exactly: "Not mentioned in the policy."
# # # # # # 4. Do NOT guess or assume anything. Only answer from the context.
# # # # # # 5. Each answer must be a single, complete, professional sentence.
# # # # # # """

# # # # # # # ========== DOC EXTRACTORS ==========
# # # # # # def extract_text_from_pdf(pdf_url: str) -> str:
# # # # # #     response = requests.get(pdf_url)
# # # # # #     response.raise_for_status()
# # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# # # # # #         tmp.write(response.content)
# # # # # #         tmp_path = tmp.name
# # # # # #     text = ""
# # # # # #     doc = fitz.open(tmp_path)
# # # # # #     for page in doc:
# # # # # #         text += page.get_text("text")
# # # # # #     doc.close()
# # # # # #     os.remove(tmp_path)
# # # # # #     return text.strip()

# # # # # # def extract_text_from_docx(docx_url: str) -> str:
# # # # # #     response = requests.get(docx_url)
# # # # # #     response.raise_for_status()
# # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# # # # # #         tmp.write(response.content)
# # # # # #         tmp_path = tmp.name
# # # # # #     doc = docx.Document(tmp_path)
# # # # # #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# # # # # #     os.remove(tmp_path)
# # # # # #     return text

# # # # # # def extract_text_from_email(email_url: str) -> str:
# # # # # #     response = requests.get(email_url)
# # # # # #     response.raise_for_status()
# # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # # # # #         tmp.write(response.content)
# # # # # #         tmp_path = tmp.name
# # # # # #     with open(tmp_path, 'rb') as f:
# # # # # #         msg = BytesParser(policy=policy.default).parse(f)
# # # # # #     os.remove(tmp_path)
# # # # # #     email_text = f"Subject: {msg['subject']}\n\n"
# # # # # #     if msg.is_multipart():
# # # # # #         for part in msg.walk():
# # # # # #             if part.get_content_type() == 'text/plain':
# # # # # #                 email_text += part.get_content()
# # # # # #     else:
# # # # # #         email_text += msg.get_content()
# # # # # #     return email_text.strip()

# # # # # # # ========== SMART CONTEXT CHUNKING ==========
# # # # # # def find_relevant_context(full_text, question, max_tokens=1800):
# # # # # #     """Finds the most relevant paragraphs for the question (simple version)"""
# # # # # #     # Heuristic: Use the keyword from the question
# # # # # #     import re
# # # # # #     qwords = [
# # # # # #         w.lower() for w in question.replace("?", "").replace(".", "").split()
# # # # # #         if len(w) > 3
# # # # # #     ]
# # # # # #     para_list = full_text.split('\n')
# # # # # #     candidates = []
# # # # # #     for para in para_list:
# # # # # #         for w in qwords:
# # # # # #             if w in para.lower():
# # # # # #                 candidates.append(para.strip())
# # # # # #                 break
# # # # # #     # Deduplicate and join
# # # # # #     context = "\n".join(list(dict.fromkeys(candidates)))
# # # # # #     if not context:
# # # # # #         # fallback to first N chars if nothing found
# # # # # #         context = full_text[:4000]
# # # # # #     # truncate to N tokens/chars
# # # # # #     return context[:6000]

# # # # # # # ========== OPENAI LLM CALL ==========
# # # # # # def call_openai_api(prompt: str, model: str, api_key: str) -> str:
# # # # # #     openai.api_key = api_key
# # # # # #     try:
# # # # # #         response = openai.chat.completions.create(
# # # # # #             model=model,
# # # # # #             messages=[{"role": "user", "content": prompt}],
# # # # # #             temperature=0.0,
# # # # # #             max_tokens=500,
# # # # # #             top_p=1,
# # # # # #         )
# # # # # #         return response.choices[0].message.content.strip()
# # # # # #     except Exception as e:
# # # # # #         raise RuntimeError(str(e))

# # # # # # @app.get("/")
# # # # # # def read_root():
# # # # # #     return {"message": "Flexible OpenAI Doc QA is running", "priority_models": OPENAI_MODELS}

# # # # # # @app.post("/api/v1/hackrx/run")
# # # # # # def run_analysis(request: RunRequest, authorization: str = Header(...)):
# # # # # #     if authorization != f"Bearer {API_TOKEN}":
# # # # # #         raise HTTPException(status_code=401, detail="Unauthorized")
# # # # # #     try:
# # # # # #         # -------- Extract content --------
# # # # # #         combined_text = ""
# # # # # #         doc_urls = []
# # # # # #         if request.documents:
# # # # # #             if isinstance(request.documents, str):
# # # # # #                 doc_urls = [request.documents]
# # # # # #             elif isinstance(request.documents, list):
# # # # # #                 doc_urls = request.documents
# # # # # #         for doc_url in doc_urls:
# # # # # #             parsed_url = urllib.parse.urlparse(doc_url)
# # # # # #             file_name = os.path.basename(parsed_url.path).lower()
# # # # # #             if file_name.endswith(".pdf"):
# # # # # #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# # # # # #             elif file_name.endswith(".docx"):
# # # # # #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# # # # # #             elif file_name.endswith(".txt"):
# # # # # #                 combined_text += "\n" + requests.get(doc_url).text
# # # # # #             else:
# # # # # #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")
# # # # # #         if request.email_file:
# # # # # #             combined_text += "\n" + extract_text_from_email(request.email_file)
# # # # # #         if not combined_text.strip():
# # # # # #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

# # # # # #         # -------- For each question, get relevant chunk --------
# # # # # #         answers = []
# # # # # #         for question in request.questions:
# # # # # #             context = find_relevant_context(combined_text, question)
# # # # # #             prompt = PROMPT_TEMPLATE.format(context=context, query=question)
# # # # # #             # -------- Switch model here --------
# # # # # #             # Try all API KEYS and MODELS (highest to lowest priority)
# # # # # #             answer = None
# # # # # #             last_exc = None
# # # # # #             for api_key in OPENAI_API_KEYS:
# # # # # #                 if not api_key:
# # # # # #                     continue
# # # # # #                 for model in OPENAI_MODELS:
# # # # # #                     try:
# # # # # #                         answer = call_openai_api(prompt, model, api_key)
# # # # # #                         break
# # # # # #                     except Exception as e:
# # # # # #                         last_exc = e
# # # # # #                         continue
# # # # # #                 if answer: break
# # # # # #             if not answer:
# # # # # #                 # fallback
# # # # # #                 answer = f"Not answered due to error: {last_exc}"
# # # # # #             answers.append(answer.strip())

# # # # # #         return {"answers": answers}

# # # # # #     except Exception as e:
# # # # # #         raise HTTPException(status_code=500, detail=str(e))




# # # # # from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File
# # # # # from pydantic import BaseModel
# # # # # import os, requests, fitz, tempfile
# # # # # from dotenv import load_dotenv
# # # # # from typing import List, Optional
# # # # # from email import policy
# # # # # from email.parser import BytesParser
# # # # # import docx
# # # # # # from langchain_community.vectorstores import FAISS
# # # # # from typing import List, Optional, Union
# # # # # from pydantic import BaseModel
# # # # # import urllib.parse

# # # # # load_dotenv()

# # # # # app = FastAPI()

# # # # # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# # # # # API_TOKEN = os.getenv("API_TOKEN")

# # # # # # Request schema
# # # # # class RunRequest(BaseModel):
# # # # #     documents: Optional[Union[str, List[str]]] = None  # URLs for docs
# # # # #     questions: List[str]
# # # # #     email_file: Optional[str] = None       # Optional email file URL

# # # # # # Prompt Template
# # # # # PROMPT_TEMPLATE = """
# # # # # You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

# # # # # Context:
# # # # # {context}

# # # # # Question:
# # # # # {query}

# # # # # Instructions:
# # # # # 1. Provide a clear and direct answer based ONLY on the context.
# # # # # 2. Do not specify clause numbers or descriptions.
# # # # # 3. If the answer is "Yes" or "No," include a short explanation based on the clause.
# # # # # 4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
# # # # # 5. Do NOT invent or assume any information outside the given context.
# # # # # 6. Limit each answer to a maximum of one paragraph.
# # # # # 7. If the context is too long, summarize it to focus on relevant parts.

# # # # # Answer:
# # # # # """

# # # # # # ---------------- Mistral API ----------------
# # # # # def call_mistral(prompt: str) -> str:
# # # # #     url = "https://api.mistral.ai/v1/chat/completions"
# # # # #     headers = {
# # # # #         "Authorization": f"Bearer {MISTRAL_API_KEY}",
# # # # #         "Content-Type": "application/json"
# # # # #     }
# # # # #     payload = {
# # # # #         "model": "mistral-small-latest",
# # # # #         "temperature": 0.3,
# # # # #         "top_p": 1,
# # # # #         "max_tokens": 500,
# # # # #         "messages": [{"role": "user", "content": prompt}]
# # # # #     }
# # # # #     res = requests.post(url, headers=headers, json=payload)
# # # # #     res.raise_for_status()
# # # # #     return res.json()["choices"][0]["message"]["content"]

# # # # # # ---------------- Document Extractors ----------------
# # # # # def extract_text_from_pdf(pdf_url: str) -> str:
# # # # #     response = requests.get(pdf_url)
# # # # #     response.raise_for_status()
# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# # # # #         tmp.write(response.content)
# # # # #         tmp_path = tmp.name

# # # # #     text = ""
# # # # #     doc = fitz.open(tmp_path)
# # # # #     for page in doc:
# # # # #         text += page.get_text("text")
# # # # #     doc.close()
# # # # #     os.remove(tmp_path)
# # # # #     return text.strip()

# # # # # def extract_text_from_docx(docx_url: str) -> str:
# # # # #     response = requests.get(docx_url)
# # # # #     response.raise_for_status()
# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# # # # #         tmp.write(response.content)
# # # # #         tmp_path = tmp.name

# # # # #     doc = docx.Document(tmp_path)
# # # # #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# # # # #     os.remove(tmp_path)
# # # # #     return text

# # # # # def extract_text_from_email(email_url: str) -> str:
# # # # #     response = requests.get(email_url)
# # # # #     response.raise_for_status()
# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # # # #         tmp.write(response.content)
# # # # #         tmp_path = tmp.name

# # # # #     with open(tmp_path, 'rb') as f:
# # # # #         msg = BytesParser(policy=policy.default).parse(f)
# # # # #     os.remove(tmp_path)

# # # # #     # Extract email text
# # # # #     email_text = f"Subject: {msg['subject']}\n\n"
# # # # #     if msg.is_multipart():
# # # # #         for part in msg.walk():
# # # # #             if part.get_content_type() == 'text/plain':
# # # # #                 email_text += part.get_content()
# # # # #     else:
# # # # #         email_text += msg.get_content()
# # # # #     return email_text.strip()

# # # # # @app.get("/")
# # # # # def read_root():
# # # # #     return {"message": "FastAPI is running"}

# # # # # # ---------------- API Endpoint ----------------
# # # # # @app.post("/api/v1/hackrx/run")
# # # # # def run_analysis(request: RunRequest, authorization: str = Header(...)):
# # # # #     if authorization != f"Bearer {API_TOKEN}":
# # # # #         raise HTTPException(status_code=401, detail="Unauthorized")

# # # # #     try:
# # # # #         combined_text = ""

# # # # #         # Extract from documents
# # # # #         doc_urls = []
# # # # #         if request.documents:
# # # # #             if isinstance(request.documents, str):
# # # # #                 doc_urls = [request.documents]
# # # # #             elif isinstance(request.documents, list):
# # # # #                 doc_urls = request.documents

# # # # #         for doc_url in doc_urls:
# # # # #             parsed_url = urllib.parse.urlparse(doc_url)
# # # # #             file_name = os.path.basename(parsed_url.path).lower()
# # # # #             if file_name.endswith(".pdf"):
# # # # #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# # # # #             elif file_name.endswith(".docx"):
# # # # #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# # # # #             elif file_name.endswith(".txt"):
# # # # #                 combined_text += "\n" + requests.get(doc_url).text
# # # # #             else:
# # # # #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")

# # # # #         # Extract from email if provided
# # # # #         if request.email_file:
# # # # #             combined_text += "\n" + extract_text_from_email(request.email_file)

# # # # #         if not combined_text.strip():
# # # # #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

# # # # #         context = combined_text.strip()

# # # # #         # Format multiple questions as a single multi-question prompt
# # # # #         numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])
# # # # #         multi_question_prompt = PROMPT_TEMPLATE.format(context=context, query=numbered_questions)
# # # # #         multi_answer_response = call_mistral(multi_question_prompt)

# # # # #         # Try splitting response by question number
# # # # #         split_answers = []
# # # # #         for i in range(len(request.questions)):
# # # # #             prefix = f"{i+1}."
# # # # #             next_prefix = f"{i+2}."
# # # # #             start = multi_answer_response.find(prefix)
# # # # #             end = multi_answer_response.find(next_prefix)
# # # # #             if start != -1:
# # # # #                 answer = multi_answer_response[start:end].strip()
# # # # #                 answer = answer.lstrip(f"{prefix}").strip()
# # # # #                 split_answers.append(answer)
        
# # # # #         # Fallback if not split properly
# # # # #         if not split_answers:
# # # # #             split_answers = [multi_answer_response.strip()]

# # # # #         return {"answers": split_answers}

# # # # #     except Exception as e:
# # # # #         raise HTTPException(status_code=500, detail=str(e))








# # # # from fastapi import FastAPI, Request, HTTPException, Header, UploadFile, File
# # # # from pydantic import BaseModel
# # # # import os, requests, fitz, tempfile
# # # # from dotenv import load_dotenv
# # # # from typing import List, Optional
# # # # from email import policy
# # # # from email.parser import BytesParser
# # # # import docx
# # # # # from langchain_community.vectorstores import FAISS
# # # # from typing import List, Optional, Union
# # # # from pydantic import BaseModel
# # # # import urllib.parse

# # # # load_dotenv()

# # # # app = FastAPI()

# # # # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY1")
# # # # API_TOKEN = os.getenv("API_TOKEN")

# # # # # Request schema
# # # # class RunRequest(BaseModel):
# # # #     documents: Optional[Union[str, List[str]]] = None  # URLs for docs
# # # #     questions: List[str]
# # # #     email_file: Optional[str] = None       # Optional email file URL

# # # # # Prompt Template
# # # # PROMPT_TEMPLATE = """
# # # # You are an insurance policy expert. Use ONLY the information provided in the context to answer the question.

# # # # Context:
# # # # {context}

# # # # Question:
# # # # {query}

# # # # Instructions:
# # # # 1. Provide a clear and direct answer based ONLY on the context.
# # # # 2. Do not specify clause numbers or descriptions.
# # # # 3. If the answer is "Yes" or "No," include a short explanation based on the clause.
# # # # 4. If the information is not found in the context, reply exactly with: "Not mentioned in the policy."
# # # # 5. Do NOT invent or assume any information outside the given context.
# # # # 6. Limit each answer to a maximum of one paragraph.
# # # # 7. If the context is too long, summarize it to focus on relevant parts.

# # # # Answer:
# # # # """

# # # # # ---------------- Mistral API ----------------
# # # # def call_mistral(prompt: str) -> str:
# # # #     url = "https://api.mistral.ai/v1/chat/completions"
# # # #     headers = {
# # # #         "Authorization": f"Bearer {MISTRAL_API_KEY}",
# # # #         "Content-Type": "application/json"
# # # #     }
# # # #     payload = {
# # # #         "model": "mistral-small-latest",
# # # #         "temperature": 0.3,
# # # #         "top_p": 1,
# # # #         "max_tokens": 500,
# # # #         "messages": [{"role": "user", "content": prompt}]
# # # #     }
# # # #     res = requests.post(url, headers=headers, json=payload)
# # # #     res.raise_for_status()
# # # #     return res.json()["choices"][0]["message"]["content"]

# # # # # ---------------- Document Extractors ----------------
# # # # def extract_text_from_pdf(pdf_url: str) -> str:
# # # #     response = requests.get(pdf_url)
# # # #     response.raise_for_status()
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# # # #         tmp.write(response.content)
# # # #         tmp_path = tmp.name

# # # #     text = ""
# # # #     doc = fitz.open(tmp_path)
# # # #     for page in doc:
# # # #         text += page.get_text("text")
# # # #     doc.close()
# # # #     os.remove(tmp_path)
# # # #     return text.strip()

# # # # def extract_text_from_docx(docx_url: str) -> str:
# # # #     response = requests.get(docx_url)
# # # #     response.raise_for_status()
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# # # #         tmp.write(response.content)
# # # #         tmp_path = tmp.name

# # # #     doc = docx.Document(tmp_path)
# # # #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# # # #     os.remove(tmp_path)
# # # #     return text

# # # # def extract_text_from_email(email_url: str) -> str:
# # # #     response = requests.get(email_url)
# # # #     response.raise_for_status()
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # # #         tmp.write(response.content)
# # # #         tmp_path = tmp.name

# # # #     with open(tmp_path, 'rb') as f:
# # # #         msg = BytesParser(policy=policy.default).parse(f)
# # # #     os.remove(tmp_path)

# # # #     # Extract email text
# # # #     email_text = f"Subject: {msg['subject']}\n\n"
# # # #     if msg.is_multipart():
# # # #         for part in msg.walk():
# # # #             if part.get_content_type() == 'text/plain':
# # # #                 email_text += part.get_content()
# # # #     else:
# # # #         email_text += msg.get_content()
# # # #     return email_text.strip()

# # # # @app.get("/")
# # # # def read_root():
# # # #     return {"message": "FastAPI is running"}

# # # # # ---------------- API Endpoint ----------------
# # # # @app.post("/api/v1/hackrx/run")
# # # # def run_analysis(request: RunRequest, authorization: str = Header(...)):
# # # #     if authorization != f"Bearer {API_TOKEN}":
# # # #         raise HTTPException(status_code=401, detail="Unauthorized")

# # # #     try:
# # # #         combined_text = ""

# # # #         # Extract from documents
# # # #         doc_urls = []
# # # #         if request.documents:
# # # #             if isinstance(request.documents, str):
# # # #                 doc_urls = [request.documents]
# # # #             elif isinstance(request.documents, list):
# # # #                 doc_urls = request.documents

# # # #         for doc_url in doc_urls:
# # # #             parsed_url = urllib.parse.urlparse(doc_url)
# # # #             file_name = os.path.basename(parsed_url.path).lower()
# # # #             if file_name.endswith(".pdf"):
# # # #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# # # #             elif file_name.endswith(".docx"):
# # # #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# # # #             elif file_name.endswith(".txt"):
# # # #                 combined_text += "\n" + requests.get(doc_url).text
# # # #             else:
# # # #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")

# # # #         # Extract from email if provided
# # # #         if request.email_file:
# # # #             combined_text += "\n" + extract_text_from_email(request.email_file)

# # # #         if not combined_text.strip():
# # # #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")

# # # #         context = combined_text.strip()

# # # #         # Format multiple questions as a single multi-question prompt
# # # #         numbered_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(request.questions)])
# # # #         multi_question_prompt = PROMPT_TEMPLATE.format(context=context, query=numbered_questions)
# # # #         multi_answer_response = call_mistral(multi_question_prompt)

# # # #         # Try splitting response by question number
# # # #         split_answers = []
# # # #         for i in range(len(request.questions)):
# # # #             prefix = f"{i+1}."
# # # #             next_prefix = f"{i+2}."
# # # #             start = multi_answer_response.find(prefix)
# # # #             end = multi_answer_response.find(next_prefix)
# # # #             if start != -1:
# # # #                 answer = multi_answer_response[start:end].strip()
# # # #                 answer = answer.lstrip(f"{prefix}").strip()
# # # #                 split_answers.append(answer)
        
# # # #         # Fallback if not split properly
# # # #         if not split_answers:
# # # #             split_answers = [multi_answer_response.strip()]

# # # #         return {"answers": split_answers}

# # # #     except Exception as e:
# # # #         raise HTTPException(status_code=500, detail=str(e))







# # # #######


# # # from fastapi import FastAPI, HTTPException, Header
# # # from pydantic import BaseModel
# # # import os, requests, fitz, tempfile, docx, urllib.parse
# # # from dotenv import load_dotenv
# # # from typing import List, Optional, Union
# # # from email import policy
# # # from email.parser import BytesParser

# # # import time

# # # load_dotenv()
# # # app = FastAPI()

# # # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# # # API_TOKEN = os.getenv("API_TOKEN")

# # # # Enhanced PROMPT_TEMPLATE
# # # PROMPT_TEMPLATE = """
# # # You are an expert insurance policy analyst. Your job is to answer ONE question at a time, ONLY using the text from the context below. 

# # # Instructions:
# # # - For the question below, return the answer **EXACTLY as stated in the context**, quoting all relevant numbers, time periods, percentages, benefit limits, eligibility criteria, exclusions, waiting periods, sub-limits, and conditions word-for-word.
# # # - Do not skip or summarize any information. **Include every detail and clause found in the context** that answers the question.
# # # - If the answer is “Yes” or “No,” always give the full policy conditions, criteria, exclusions, or requirements following your answer.
# # # - If the answer refers to a definition (like ‘Hospital’ or ‘AYUSH’), copy the full formal definition as given in the context.
# # # - If information is not found in the policy, reply exactly: "Not mentioned in the policy."
# # # - **Do not invent or infer anything not directly found in the context.**
# # # - Begin your answer with: “{question_number}. ” (example: “3. Yes, the policy covers ...”)
# # # - Write your answer as a single, complete, formal sentence (or sentences), matching the tone of the policy.

# # # Context:
# # # {context}

# # # Question:
# # # {question}
# # # """

# # # class RunRequest(BaseModel):
# # #     documents: Optional[Union[str, List[str]]] = None
# # #     questions: List[str]
# # #     email_file: Optional[str] = None

# # # def call_mistral(prompt: str) -> str:
# # #     url = "https://api.mistral.ai/v1/chat/completions"
# # #     headers = {
# # #         "Authorization": f"Bearer {MISTRAL_API_KEY}",
# # #         "Content-Type": "application/json"
# # #     }
# # #     payload = {
# # #         "model": "mistral-small-latest",
# # #         "temperature": 0.3,
# # #         "top_p": 1,
# # #         "max_tokens": 512,
# # #         "messages": [{"role": "user", "content": prompt}]
# # #     }
# # #     res = requests.post(url, headers=headers, json=payload)
# # #     res.raise_for_status()
# # #     return res.json()["choices"][0]["message"]["content"]

# # # def extract_text_from_pdf(pdf_url: str) -> str:
# # #     response = requests.get(pdf_url)
# # #     response.raise_for_status()
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# # #         tmp.write(response.content)
# # #         tmp_path = tmp.name
# # #     text = ""
# # #     doc = fitz.open(tmp_path)
# # #     for page in doc:
# # #         text += page.get_text("text")
# # #     doc.close()
# # #     os.remove(tmp_path)
# # #     return text.strip()

# # # def extract_text_from_docx(docx_url: str) -> str:
# # #     response = requests.get(docx_url)
# # #     response.raise_for_status()
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# # #         tmp.write(response.content)
# # #         tmp_path = tmp.name
# # #     doc = docx.Document(tmp_path)
# # #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# # #     os.remove(tmp_path)
# # #     return text

# # # def extract_text_from_email(email_url: str) -> str:
# # #     response = requests.get(email_url)
# # #     response.raise_for_status()
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# # #         tmp.write(response.content)
# # #         tmp_path = tmp.name
# # #     with open(tmp_path, 'rb') as f:
# # #         msg = BytesParser(policy=policy.default).parse(f)
# # #     os.remove(tmp_path)
# # #     email_text = f"Subject: {msg['subject']}\n\n"
# # #     if msg.is_multipart():
# # #         for part in msg.walk():
# # #             if part.get_content_type() == 'text/plain':
# # #                 email_text += part.get_content()
# # #     else:
# # #         email_text += msg.get_content()
# # #     return email_text.strip()

# # # @app.get("/")
# # # def read_root():
# # #     return {"message": "FastAPI with strict per-question Mistral prompt"}

# # # @app.post("/api/v1/hackrx/run")
# # # def run_analysis(request: RunRequest, authorization: str = Header(...)):
# # #     if authorization != f"Bearer {API_TOKEN}":
# # #         raise HTTPException(status_code=401, detail="Unauthorized")

# # #     try:
# # #         combined_text = ""
# # #         doc_urls = []
# # #         if request.documents:
# # #             if isinstance(request.documents, str):
# # #                 doc_urls = [request.documents]
# # #             elif isinstance(request.documents, list):
# # #                 doc_urls = request.documents
# # #         for doc_url in doc_urls:
# # #             parsed_url = urllib.parse.urlparse(doc_url)
# # #             file_name = os.path.basename(parsed_url.path).lower()
# # #             if file_name.endswith(".pdf"):
# # #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# # #             elif file_name.endswith(".docx"):
# # #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# # #             elif file_name.endswith(".txt"):
# # #                 combined_text += "\n" + requests.get(doc_url).text
# # #             else:
# # #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")
# # #         if request.email_file:
# # #             combined_text += "\n" + extract_text_from_email(request.email_file)
# # #         if not combined_text.strip():
# # #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")
# # #         context = combined_text.strip()
        
# # #         # Per-question strict answering with sequence/index
# # #         answers = []
# # #         for idx, q in enumerate(request.questions):
# # #             question_number = idx + 1
# # #             single_prompt = PROMPT_TEMPLATE.format(
# # #                 context=context,
# # #                 question=q,
# # #                 question_number=question_number
# # #             )
# # #             response = call_mistral(single_prompt)
# # #             # Remove question number prefix for JSON if you want, or keep for clarity
# # #             ans = response.strip()
# # #             # Optionally, remove the "1. " prefix
# # #             if ans.startswith(f"{question_number}."):
# # #                 ans = ans[len(f"{question_number}."):].strip()
# # #             answers.append(ans)
# # #             time.sleep(0.5)  # Be nice to the API

# # #         return {"answers": answers}

# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=str(e))


# # ###### temprature = 0.3 accu increase = 


# # from fastapi import FastAPI, HTTPException, Header
# # from pydantic import BaseModel
# # import os, requests, fitz, tempfile, docx, urllib.parse
# # from dotenv import load_dotenv
# # from typing import List, Optional, Union
# # from email import policy
# # from email.parser import BytesParser
# # import httpx
# # import asyncio

# # load_dotenv()
# # app = FastAPI()



# # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# # API_TOKEN = os.getenv("API_TOKEN")

# # # Enhanced PROMPT_TEMPLATE (unchanged, as you requested)
# # PROMPT_TEMPLATE = """
# # You are an expert insurance policy analyst. Your job is to answer ONE question at a time, ONLY using the text from the context below. 

# # Instructions:
# # - For the question below, return the answer **EXACTLY as stated in the context**, quoting all relevant numbers, time periods, percentages, benefit limits, eligibility criteria, exclusions, waiting periods, sub-limits, and conditions word-for-word.
# # - Do not skip or summarize any information. **Include every detail and clause found in the context** that answers the question.
# # - If the answer is “Yes” or “No,” always give the full policy conditions, criteria, exclusions, or requirements following your answer.
# # - If the answer refers to a definition (like ‘Hospital’ or ‘AYUSH’), copy the full formal definition as given in the context.
# # - If information is not found in the policy, reply exactly: "Not mentioned in the policy."
# # - **Do not invent or infer anything not directly found in the context.**
# # - Begin your answer with: “{question_number}. ” (example: “3. Yes, the policy covers ...”)
# # - Write your answer as a single, complete, formal sentence (or sentences), matching the tone of the policy.

# # Context:
# # {context}

# # Question:
# # {question}
# # """

# # class RunRequest(BaseModel):
# #     documents: Optional[Union[str, List[str]]] = None
# #     questions: List[str]
# #     email_file: Optional[str] = None

# # async def call_mistral_async(prompt: str) -> str:
# #     url = "https://api.mistral.ai/v1/chat/completions"
# #     headers = {
# #         "Authorization": f"Bearer {MISTRAL_API_KEY}",
# #         "Content-Type": "application/json"
# #     }
# #     payload = {
# #         "model": "mistral-small-latest",
# #         "temperature": 0,
# #         "top_p": 1,
# #         "max_tokens": 512,
# #         "messages": [{"role": "user", "content": prompt}]
# #     }
# #     async with httpx.AsyncClient(timeout=60) as client:
# #         res = await client.post(url, headers=headers, json=payload)
# #         res.raise_for_status()
# #         return res.json()["choices"][0]["message"]["content"]

# # def extract_text_from_pdf(pdf_url: str) -> str:
# #     response = requests.get(pdf_url)
# #     response.raise_for_status()
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
# #         tmp.write(response.content)
# #         tmp_path = tmp.name
# #     text = ""
# #     doc = fitz.open(tmp_path)
# #     for page in doc:
# #         text += page.get_text("text")
# #     doc.close()
# #     os.remove(tmp_path)
# #     return text.strip()

# # def extract_text_from_docx(docx_url: str) -> str:
# #     response = requests.get(docx_url)
# #     response.raise_for_status()
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
# #         tmp.write(response.content)
# #         tmp_path = tmp.name
# #     doc = docx.Document(tmp_path)
# #     text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
# #     os.remove(tmp_path)
# #     return text

# # def extract_text_from_email(email_url: str) -> str:
# #     response = requests.get(email_url)
# #     response.raise_for_status()
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=".eml") as tmp:
# #         tmp.write(response.content)
# #         tmp_path = tmp.name
# #     with open(tmp_path, 'rb') as f:
# #         msg = BytesParser(policy=policy.default).parse(f)
# #     os.remove(tmp_path)
# #     email_text = f"Subject: {msg['subject']}\n\n"
# #     if msg.is_multipart():
# #         for part in msg.walk():
# #             if part.get_content_type() == 'text/plain':
# #                 email_text += part.get_content()
# #     else:
# #         email_text += msg.get_content()
# #     return email_text.strip()

# # @app.get("/")
# # def read_root():
# #     return {"message": "FastAPI with strict per-question Mistral prompt (async parallel version)"}

# # @app.post("/api/v1/hackrx/run")
# # async def run_analysis(request: RunRequest, authorization: str = Header(...)):
# #     if authorization != f"Bearer {API_TOKEN}":
# #         raise HTTPException(status_code=401, detail="Unauthorized")
# #     try:
# #         combined_text = ""
# #         doc_urls = []
# #         if request.documents:
# #             if isinstance(request.documents, str):
# #                 doc_urls = [request.documents]
# #             elif isinstance(request.documents, list):
# #                 doc_urls = request.documents
# #         for doc_url in doc_urls:
# #             parsed_url = urllib.parse.urlparse(doc_url)
# #             file_name = os.path.basename(parsed_url.path).lower()
# #             if file_name.endswith(".pdf"):
# #                 combined_text += "\n" + extract_text_from_pdf(doc_url)
# #             elif file_name.endswith(".docx"):
# #                 combined_text += "\n" + extract_text_from_docx(doc_url)
# #             elif file_name.endswith(".txt"):
# #                 combined_text += "\n" + requests.get(doc_url).text
# #             else:
# #                 raise HTTPException(status_code=400, detail=f"Unsupported document format: {doc_url}")
# #         if request.email_file:
# #             combined_text += "\n" + extract_text_from_email(request.email_file)
# #         if not combined_text.strip():
# #             raise HTTPException(status_code=400, detail="No valid content extracted from provided sources.")
# #         context = combined_text.strip()
        
# #         async def get_answer(idx, q):
# #             question_number = idx + 1
# #             single_prompt = PROMPT_TEMPLATE.format(
# #                 context=context,
# #                 question=q,
# #                 question_number=question_number
# #             )
# #             resp = await call_mistral_async(single_prompt)
# #             ans = resp.strip()
# #             if ans.startswith(f"{question_number}."):
# #                 ans = ans[len(f"{question_number}."):].strip()
# #             return ans
        
# #         tasks = [get_answer(idx, q) for idx, q in enumerate(request.questions)]
# #         answers = await asyncio.gather(*tasks)
# #         return {"answers": answers}

# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=str(e))




# ##################  6 pdf run  code 


# # api.py — FastAPI with strict per-question Mistral→Groq fallback + robust PDF chunking + selective OCR + safe OCR fallback
# # pip install fastapi uvicorn python-dotenv httpx pymupdf pillow pytesseract python-docx rapidfuzz
# # (Windows: install Tesseract OCR: https://github.com/tesseract-ocr/tesseract)
# # export API_TOKEN=… MISTRAL_API_KEY=… GROQ_API_KEY=…
# # uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# from fastapi import FastAPI, HTTPException, Header
# from pydantic import BaseModel
# import os, requests, fitz, tempfile, docx, urllib.parse, io
# from dotenv import load_dotenv
# from typing import List, Optional, Union
# from email import policy
# from email.parser import BytesParser
# import httpx, asyncio

# from PIL import Image
# import pytesseract
# from rapidfuzz import fuzz

# load_dotenv()
# app = FastAPI()

# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# API_TOKEN       = os.getenv("API_TOKEN")
# GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

# PROMPT_TEMPLATE = """
# You are an expert insurance policy analyst. Your job is to answer ONE question at a time, ONLY using the text from the context below. 

# Instructions:
# - For the question below, return the answer **EXACTLY as stated in the context**, quoting all relevant numbers, time periods, percentages, benefit limits, eligibility criteria, exclusions, waiting periods, sub-limits, and conditions word‐for‐word.
# - Do not skip or summarize any information. **Include every detail and clause found in the context** that answers the question.
# - If additional explanation is present in nearby sentences or sections, include those as well to make the answer more complete.
# - If the answer is “Yes” or “No,” always give the full policy conditions, criteria, exclusions, or requirements following your answer.
# - If the answer refers to a definition (like ‘Hospital’ or ‘AYUSH’), copy the full formal definition as given in the context.
# - If information is not found in the policy, reply exactly: "Not mentioned in the policy."
# - **Do not invent or infer anything not directly found in the context.**
# - Begin your answer with: “{question_number}. ” (example: “3. Yes, the policy covers ...”)
# - Write your answer as a single, complete, formal sentence (or sentences), matching the tone of the policy.

# Context:
# {context}

# Question:
# {question}
# """

# class RunRequest(BaseModel):
#     documents: Optional[Union[str, List[str]]] = None
#     questions: List[str]
#     email_file: Optional[str] = None

# # --- Groq fallback ---
# async def call_groq_async(prompt: str) -> str:
#     if not GROQ_API_KEY:
#         raise HTTPException(status_code=502, detail="Groq fallback unavailable (GROQ_API_KEY not set)")
#     url = "https://api.groq.com/openai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type":  "application/json"
#     }
#     payload = {
#         "model":       "llama3-70b-8192",
#         "temperature": 0,
#         "top_p":       1,
#         "max_tokens":  850,
#         "messages":    [{"role": "user", "content": prompt}]
#     }
#     async with httpx.AsyncClient(timeout=60) as client:
#         res = await client.post(url, headers=headers, json=payload)
#         res.raise_for_status()
#         return res.json()["choices"][0]["message"]["content"]

# # --- Mistral primary + fallback ---
# async def call_mistral_async(prompt: str) -> str:
#     url = "https://api.mistral.ai/v1/chat/completions"
#     headers = {
#         "Authorization": f"Bearer {MISTRAL_API_KEY}",
#         "Content-Type":  "application/json"
#     }
#     payload = {
#         "model":       "mistral-small-latest",
#         "temperature": 0,
#         "top_p":       1,
#         "max_tokens":  850,
#         "messages":    [{"role": "user", "content": prompt}]
#     }
#     async with httpx.AsyncClient(timeout=60) as client:
#         res = await client.post(url, headers=headers, json=payload)
#         if res.status_code in (429, 413, 500, 502, 503) and GROQ_API_KEY:
#             return await call_groq_async(prompt)
#         res.raise_for_status()
#         return res.json()["choices"][0]["message"]["content"]

# # --- PDF extraction w/ per-page text/OCR (handles 400+ pages, chunking for LLM context size) ---
# MIN_TEXT_LEN = 50  # pages shorter than this run through OCR

# def extract_text_pages_from_pdf(pdf_url: str) -> list:
#     r = requests.get(pdf_url)
#     r.raise_for_status()
#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
#         tmp.write(r.content)
#         path = tmp.name

#     page_texts = []
#     doc = fitz.open(path)
#     for i, page in enumerate(doc):
#         txt = page.get_text("text") or ""
#         if len(txt.strip()) < MIN_TEXT_LEN:
#             # Try OCR, handle all exceptions gracefully
#             try:
#                 pix = page.get_pixmap(dpi=200)
#                 img = Image.open(io.BytesIO(pix.tobytes("png")))
#                 try:
#                     ocr_txt = pytesseract.image_to_string(img)
#                     if len(ocr_txt.strip()) > len(txt.strip()):
#                         print(f"[OCR] Page {i+1}: OCR used.")
#                         txt = ocr_txt
#                     else:
#                         print(f"[OCR] Page {i+1}: OCR produced less text, using original.")
#                 except pytesseract.TesseractNotFoundError as e:
#                     print(f"[OCR ERROR] Tesseract not found: {e}. Skipping OCR for page {i+1}.")
#                 except Exception as e:
#                     print(f"[OCR ERROR] Other OCR error on page {i+1}: {e}")
#             except Exception as e:
#                 print(f"[PDF→Image ERROR] Could not render page {i+1} to image: {e}")
#         page_texts.append(txt)
#     doc.close()
#     os.remove(path)
#     return page_texts  # returns list of page_texts

# def find_relevant_chunk(pages, question, window=2):
#     # Find best match page and expand to window pages before/after
#     best_score = -1
#     best_idx = 0
#     for i, txt in enumerate(pages):
#         score = fuzz.token_set_ratio(question, txt)
#         if score > best_score:
#             best_score = score
#             best_idx = i
#     # Expand window for more context, but stay in bounds
#     start = max(0, best_idx - window)
#     end   = min(len(pages), best_idx + window + 1)
#     chunk = "\n".join(pages[start:end])
#     # Limit to 8500 chars (about 3400 tokens)
#     return chunk[:8500]

# def extract_text_from_docx(docx_url: str) -> str:
#     r = requests.get(docx_url)
#     r.raise_for_status()
#     with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
#         tmp.write(r.content)
#         path = tmp.name
#     doc = docx.Document(path)
#     text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
#     os.remove(path)
#     return text

# def extract_text_from_email(email_url: str) -> str:
#     r = requests.get(email_url)
#     r.raise_for_status()
#     with tempfile.NamedTemporaryFile(suffix=".eml", delete=False) as tmp:
#         tmp.write(r.content)
#         path = tmp.name
#     with open(path, "rb") as f:
#         msg = BytesParser(policy=policy.default).parse(f)
#     os.remove(path)
#     body = f"Subject: {msg['subject']}\n\n"
#     if msg.is_multipart():
#         for part in msg.walk():
#             if part.get_content_type() == "text/plain":
#                 body += part.get_content()
#     else:
#         body += msg.get_content()
#     return body.strip()

# @app.get("/")
# def health():
#     return {"message": "OK"}

# @app.post("/api/v1/hackrx/run")
# async def run_analysis(request: RunRequest, authorization: str = Header(...)):
#     print("Received request:", request)
#     if authorization != f"Bearer {API_TOKEN}":
#         raise HTTPException(status_code=401, detail="Unauthorized")

#     docs = []
#     if request.documents:
#         docs = [request.documents] if isinstance(request.documents, str) else request.documents
#     if not docs or not request.questions:
#         raise HTTPException(status_code=400, detail="Provide both 'documents' and 'questions'.")

#     all_doc_chunks = []
#     all_doc_types  = []
#     for url in docs:
#         path = urllib.parse.urlparse(url).path.lower()
#         if path.endswith(".pdf"):
#             all_doc_chunks.append(extract_text_pages_from_pdf(url))  # List of pages
#             all_doc_types.append("pdf")
#         elif path.endswith(".docx"):
#             all_doc_chunks.append([extract_text_from_docx(url)])
#             all_doc_types.append("docx")
#         elif path.endswith(".txt"):
#             all_doc_chunks.append([requests.get(url).text])
#             all_doc_types.append("txt")
#         else:
#             raise HTTPException(400, detail=f"Unsupported format: {url}")

#     if not all_doc_chunks:
#         raise HTTPException(400, detail="No content extracted.")

#     async def get_answer(i: int, q: str) -> str:
#         # Use chunked context for each question
#         if len(all_doc_chunks) == 1 and all_doc_types[0] == "pdf":
#             pages = all_doc_chunks[0]
#             relevant_context = find_relevant_chunk(pages, q, window=2)  # get 2 pages before/after
#         else:
#             # For docx/txt/other, just use all text (truncate for LLM input)
#             relevant_context = "\n".join(c for doc in all_doc_chunks for c in doc)
#             relevant_context = relevant_context[:8500]
#         prompt = PROMPT_TEMPLATE.format(
#             context=relevant_context,
#             question=q,
#             question_number=i+1
#         )
#         resp = await call_mistral_async(prompt)
#         ans  = resp.strip()
#         prefix = f"{i+1}."
#         if ans.startswith(prefix):
#             ans = ans[len(prefix):].strip()
#         return ans

#     tasks   = [get_answer(i, q) for i, q in enumerate(request.questions)]
#     answers = await asyncio.gather(*tasks)
#     return {"answers": answers}
