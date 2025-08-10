# main.py — PDF QA API (Level-4 route + language-aware answers)
# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Env: API_TOKEN (required), MISTRAL_API_KEY (optional), GROQ_API_KEY (optional)

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Tuple, Optional
import os, tempfile, io, time, re, json, asyncio, collections, unicodedata
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import requests, httpx
from dotenv import load_dotenv
from urllib.parse import urlparse
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- App & Clients ----------------
app = FastAPI()

REQUESTS_SESSION = requests.Session()
REQUESTS_SESSION.headers.update({"Accept-Encoding": "gzip, deflate"})
REQUESTS_SESSION.mount(
    "https://",
    HTTPAdapter(
        pool_connections=20, pool_maxsize=20,
        max_retries=Retry(total=2, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504])
    ),
)

ASYNC_CLIENT = httpx.AsyncClient(
    http2=True,
    timeout=20.0,  # SUGGESTION: drop to 12–15s if you need stricter SLA
    headers={"Accept-Encoding": "gzip, deflate"},
    limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
)

@app.on_event("shutdown")
async def shutdown_event():
    await ASYNC_CLIENT.aclose()

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
API_TOKEN       = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("API_TOKEN must be set")

# ---------------- Schemas ----------------
class RunRequest(BaseModel):
    documents: str
    questions: List[str]

# ---------------- Language helpers ----------------
# ---------------- Language helpers ----------------
def is_malayalam(s: str) -> bool:
    return any(0x0D00 <= ord(ch) <= 0x0D7F for ch in (s or ""))

def question_lang_label(q: str) -> str:
    return "Malayalam" if is_malayalam(q) else "English"

def enforce_lang_instruction(q: str) -> str:
    return (
        f"IMPORTANT: Answer in the SAME language as THIS question: {question_lang_label(q)}. "
        f"Do NOT switch languages."
    )


DEFAULT_LANG_RULE = "Answer in the same language as the question."

# ---------------- Prompts ----------------
FULL_PROMPT_TEMPLATE = """You are an insurance policy expert. Use ONLY the information provided in the context to answer the questions.
Context:
{context}

Questions:
{query}

Instructions:
1. Provide clear and direct answers based ONLY on the context.
2. Do not specify the clause number or clause description.
3. If the answer is "Yes" or "No," include a short explanation.
4. If not found in the context, reply: "Not mentioned in the policy."
5. Give each answer in a single paragraph without numbering.

Answers:"""

CHUNK_PROMPT_TEMPLATE = """You are an insurance policy specialist. Prefer answers from the policy <Context>.

Decision rule:
1) Search ALL of <Context>. If the answer exists there, answer ONLY from <Context>.
2) If the answer is NOT in <Context>, reply exactly: "Not mentioned in the policy."
3) Answer in the SAME LANGUAGE as the question.
4) {lang_rule}

Requirements:
- Quote every number, amount, time period, percentage, sub-limit, definition, eligibility, exclusion, waiting period, and condition **word-for-word**.
- If any numbers or conditions are present in the answer, include at least one short **verbatim** quote from <Context>.
- If Yes/No, start with “Yes.” or “No.” and immediately quote the rule that makes it so.
- Include all applicable conditions in a compact way.
- No clause numbers, no speculation, no invented facts.

Context:
{context}

Questions:
{query}

Answers (one concise paragraph per question, no bullets, no numbering):
"""

WEB_PROMPT_TEMPLATE = """You are an expert insurance policy assistant. Based on the document titled "{title}", answer the following questions using general or public insurance knowledge.
Title: "{title}"

Questions:
{query}

Instructions:
- Use public knowledge.
- If specific document data is needed, reply: "Not found in public sources."
- Answer EACH question in the SAME LANGUAGE it is asked (English → English, Malayalam → Malayalam). Do not mix languages across answers.
- Keep each answer concise (1 paragraph max).
- Give each answer in a single paragraph without numbering.

Answers:"""

# ---------------- Helpers ----------------


def approx_tokens_from_text(s: str) -> int:
    # crude estimate: ~4 chars/token
    return max(1, len(s) // 4)

def choose_mistral_params(page_count: int, context_text: Optional[str]):
    # NOTE: keep your budget logic intact
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 100:
        max_tokens, temperature, timeout = 1200, 0.1, 18
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1300, 0.18, 15
    else:
        max_tokens, temperature, timeout = 700, 0.18, 12
    total_budget = 3500
    budget_left = max(700, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}

def choose_groq_params(page_count: int, context_text: Optional[str]):
    ctx_tok = approx_tokens_from_text(context_text or "")
    if page_count <= 100:
        max_tokens, temperature, timeout = 1300, 0.0, 30
    elif page_count <= 200:
        max_tokens, temperature, timeout = 1700, 0.2, 30
    else:
        max_tokens, temperature, timeout = 1100, 0.13, 25
    total_budget = 3500
    budget_left = max(800, total_budget - ctx_tok)
    return {"max_tokens": min(max_tokens, budget_left), "temperature": temperature, "timeout": timeout}

def make_question_block(questions: List[str]) -> str:
    return "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

# ---------------- PDF Extraction ----------------
def extract_text_from_pdf_url(pdf_url: str) -> Tuple[str, int, str]:
    r = REQUESTS_SESSION.get(pdf_url, timeout=20)
    r.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name

    full_text, title = "", ""
    with fitz.open(tmp_path) as doc:
        page_count = len(doc)

        # Title: quick try from first few pages; OCR fallback
        for i in range(min(15, page_count)):
            t = (doc[i].get_text() or "").strip()
            if not t:
                try:
                    pix = doc[i].get_pixmap(dpi=120)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    try:
                        t = pytesseract.image_to_string(img, lang="eng+mal").strip()
                        print("Malayalam OCR result:", t)
                    except Exception:
                        t = pytesseract.image_to_string(img, lang="eng").strip()
                except Exception:
                    continue
            if t:
                title = t.splitlines()[0][:100]
                break

        # Full text (<=200 pages)
        if page_count <= 200:
            for i in range(page_count):
                t = (doc[i].get_text() or "").strip()
                if not t:
                    try:
                        pix = doc[i].get_pixmap(dpi=120)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        t = pytesseract.image_to_string(img, lang="eng").strip()
                    except Exception:
                        t = ""
                if t:
                    full_text += t + "\n"

    os.remove(tmp_path)
    return (full_text.strip() if page_count <= 200 else "", page_count, title or "Untitled Document")

def split_text(text: str, chunk_size: int = 1400, overlap: int = 300) -> List[str]:
    chunks, start = [], 0
    n = len(text)
    while start < n and len(chunks) < 15:
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def _fast_title_and_snippets_with_ocr(data: bytes) -> tuple[int, str, str]:
    page_count = 0
    title = "Untitled Document"
    snippets = []
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            page_count = len(doc)
            sample_idxs = [i for i in (0, 1, page_count - 1) if 0 <= i < page_count]
            for idx in sample_idxs:
                page = doc[idx]
                txt = (page.get_text("text", sort=True) or "").strip()
                if not txt:
                    def _ocr_at_dpi(dpi: int) -> str:
                        pix = page.get_pixmap(dpi=dpi)
                        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
                        if img.width < 1500:
                            img = img.resize((int(img.width * 1.5), int(img.height * 1.5)))
                        img = img.point(lambda p: 255 if p > 180 else 0)
                        return (pytesseract.image_to_string(img, lang="eng+mal") or "").strip()
                    txt = _ocr_at_dpi(120)
                    if len(txt) < 400 or sum(ch.isdigit() for ch in txt) < 3:
                        txt = _ocr_at_dpi(160)
                if idx in (0, 1) and txt and title == "Untitled Document":
                    title = txt.splitlines()[0][:120]
                if txt:
                    snippets.append("\n".join(txt.splitlines()[:30])[:1500])
    except Exception:
        pass
    tiny = ("\n\n---\n\n").join(snippets)[:3000] if snippets else ""
    return page_count, title, tiny

# ---------------- LLM Calls ----------------

def call_mistral(prompt: str, params: dict, system: Optional[str] = None) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": "mistral-small-latest",
        "temperature": params.get("temperature", 0.3),
        "top_p": 1,
        "max_tokens": params.get("max_tokens", 1100),
        "messages": messages,
    }
    r = REQUESTS_SESSION.post(url, headers=headers, json=payload, timeout=params.get("timeout", 15))
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _score_chunk(q: str, c: str) -> int:
    WORD_RX = re.compile(r"\w+")
    NUM_RX  = re.compile(r"\d+%?")
    qt = set(WORD_RX.findall(q.lower()))
    ct = set(WORD_RX.findall(c.lower()))
    base = len(qt & ct)
    num_bonus = 2 * len(set(NUM_RX.findall(q)) & set(NUM_RX.findall(c)))
    return base + num_bonus

def _topk_chunks(q: str, chunks: List[str], k: int = 4) -> List[str]:
    scored = sorted((( _score_chunk(q, c), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]] if scored else []

KEY_LINE_RX = re.compile(
    r'(\b\d+\s*(day|days|month|months|year|years|%)\b|sub-?limit|room rent|ICU|AYUSH|grace|waiting|'
    r'deductible|co-?pay|exclusion|PED|check[-\s]?up|sum insured|premium|pre[-\s]?auth|pre[-\s]?existing)',
    re.I
)

def _harvest_numeric_lines(text: str, max_lines: int = 60) -> str:
    seen, out = set(), []
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        if KEY_LINE_RX.search(ln) and ln not in seen:
            seen.add(ln); out.append(ln)
            if len(out) >= max_lines: break
    return "\n".join(out)

def call_mistral_on_chunks(chunks: List[str], questions: List[str], params: dict) -> List[str]:
    answers = []

    # sys_msg = "Follow instructions exactly. Never change the language of the answer from the question's language."

    sys_msg = (
  "Answer ONLY from <Context>. If not present in <Context>, reply exactly: "
  "\"Not mentioned in the policy.\" Output one short paragraph, no bullets, no labels. "
  "Do NOT infer or add new facts. Answer strictly in the same language as the question."
)


    for q in questions:

        kchunks = _topk_chunks(q, chunks, k=4) or chunks[:4]
        combined = "\n\n".join(kchunks)
        evidence = _harvest_numeric_lines(combined)
        context = combined + (f"\n\n--- Evidence ---\n{evidence}" if evidence else "")
        lang_rule = enforce_lang_instruction(q)
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, web_snippets="", query=q, lang_rule=lang_rule)
        ans = call_mistral(prompt, params, system=sys_msg).strip()
        answers.append(ans)
    return answers

async def call_groq_on_chunks(chunks: List[str], questions: List[str], params: dict) -> List[str]:
    answers = []
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    url = "https://api.groq.com/openai/v1/chat/completions"
    sys_msg = "Follow instructions exactly. Never change the language of the answer from the question's language."

    async def ask(q: str):
        kchunks = _topk_chunks(q, chunks, k=4) or chunks[:4]
        combined = "\n\n".join(kchunks)
        evidence = _harvest_numeric_lines(combined)
        context = combined + (f"\n\n--- Evidence ---\n{evidence}" if evidence else "")
        lang_rule = enforce_lang_instruction(q)
        prompt = CHUNK_PROMPT_TEMPLATE.format(context=context, web_snippets="", query=q, lang_rule=lang_rule)
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": params.get("temperature", 0.3),
            "top_p": 1,
            "max_tokens": params.get("max_tokens", 1000),
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt}
            ],
        }
        r = await ASYNC_CLIENT.post(url, headers=headers, json=payload, timeout=params.get("timeout", 20))
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    results = await asyncio.gather(*[ask(q) for q in questions])
    answers.extend(results)
    return answers

# ---------------- Level-4 helpers ----------------


_NOT_FOUND_RX = re.compile(r"^\s*not\s+mentioned\s+in\s+the\s+policy\.?\s*$", re.I)

def _sanitize_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\d+[\.\)]\s*", "", s)
    s = re.sub(r"^Answer\s*:\s*", "", s, flags=re.I)
    return s.strip()

def _is_not_found(s: str) -> bool:
    return bool(_NOT_FOUND_RX.match((s or "").strip()))

def _topk_indices(q: str, chunks: List[str], k: int = 5) -> List[int]:
    scored = sorted(((i, _score_chunk(q, c)) for i, c in enumerate(chunks)), key=lambda x: x[1], reverse=True)
    return [i for i, _ in scored[:k]]

def _context_with_neighbors(chunks: List[str], idxs: List[int], neighbor: int = 1, budget_chars: int = 9000) -> str:
    want = set()
    for i in idxs:
        for j in range(max(0, i - neighbor), min(len(chunks), i + neighbor + 1)):
            want.add(j)
    ordered = sorted(want)
    buf, used = [], 0
    for j in ordered:
        seg = chunks[j].strip()
        if not seg:
            continue
        if used + len(seg) + 2 > budget_chars:
            break
        buf.append(seg)
        used += len(seg) + 2
    evidence = _harvest_numeric_lines("\n\n".join(buf), max_lines=120)
    if evidence:
        buf.append("\n--- Evidence ---\n" + evidence)
    return "\n\n".join(buf)


# --- Formatting helpers (leaderboard-friendly) ---

# --- Formatting helpers (leaderboard-friendly) ---
_MAL = r"\u0D00-\u0D7F"

def _unescape_quotes(s: str) -> str:
    s = s.replace('\\"', '"').replace("\\'", "'")
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s

def _strip_meta_lines(s: str) -> str:
    bad = [
        r"^\s*no direct quote.*$",
        r"^\s*related verbatim quote.*$",
        r"^\s*note:.*$",
        r"^\s*explanation:.*$",
    ]
    for rx in bad:
        s = re.sub(rx, "", s, flags=re.I | re.M)
    return s

def _single_paragraph(s: str) -> str:
    s = s.replace("\r", " ")
    s = re.sub(r"\s*\n\s*", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _fix_malayalam_spacing(s: str) -> str:
    s = re.sub(rf"([{_MAL}])(\d)", r"\1 \2", s)
    s = re.sub(rf"(\d)([{_MAL}])", r"\1 \2", s)
    s = re.sub(r"(\d)\s*%", r"\1 %", s)
    s = re.sub(rf"([{_MAL}])\s*(\d+)\s*-\s*([{_MAL}])", r"\1 \2-\3", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _postshape(a: str, target_lang: str) -> str:
    a = (a or "").strip()
    a = _unescape_quotes(a)
    a = _strip_meta_lines(a)
    a = _single_paragraph(a)
    if (a.startswith('"') and a.endswith('"')) or (a.startswith("'") and a.endswith("'")):
        a = a[1:-1].strip()
    a = a.rstrip(" .")
    if target_lang == "Malayalam":
        a = _fix_malayalam_spacing(a)
    return a



async def _retry_per_question(q: str, full_text: str, chunks: List[str], page_count: int) -> str:
    lang_rule = enforce_lang_instruction(q)
    sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."

    # Mistral retry
    try:
        idxs = _topk_indices(q, chunks, k=6)
        ctx = _context_with_neighbors(chunks, idxs, neighbor=1, budget_chars=9500)
        m_params = choose_mistral_params(page_count, ctx)
        a = _sanitize_line(call_mistral(
            CHUNK_PROMPT_TEMPLATE.format(context=ctx, query=q, lang_rule=lang_rule),
            m_params,
            system=sys_msg
        ))
        if a and not _is_not_found(a):
            return _postshape(a, question_lang_label(q))
    except Exception:
        pass

    # Groq fallback
    try:
        g_params = choose_groq_params(page_count, "\n".join(chunks[:10]))
        kchunks = _topk_chunks(q, chunks, k=6) or chunks[:4]
        combined = "\n\n".join(kchunks)
        evidence = _harvest_numeric_lines(combined)
        ctx2 = combined + (f"\n\n--- Evidence ---\n{evidence}" if evidence else "")
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "temperature": g_params.get("temperature", 0.3),
            "top_p": 1,
            "max_tokens": g_params.get("max_tokens", 1000),
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": CHUNK_PROMPT_TEMPLATE.format(context=ctx2, query=q, lang_rule=lang_rule)}
            ],
        }
        r = await ASYNC_CLIENT.post(url, headers=headers, json=payload, timeout=g_params.get("timeout", 20))
        r.raise_for_status()
        a2 = _sanitize_line(r.json()["choices"][0]["message"]["content"].strip())
        if a2:
            return _postshape(a2, question_lang_label(q))
    except Exception:
        pass

    return "Not mentioned in the policy."


# --- detect PDF vs mission URL ---
def _is_pdf_payload(data: bytes, ctype: str) -> bool:
    return (ctype and "pdf" in ctype.lower()) or data.startswith(b"%PDF")

def _is_mission_host(url: str) -> bool:
    try:
        return "register.hackrx.in" in urlparse(url).netloc.lower()
    except Exception:
        return False

# --- secret token extraction (hardened) ---
TOKEN_KEYS = ("secret_token", "secretToken", "token", "apiKey", "apikey", "key", "secret")

def _extract_token_from_json(js):
    if isinstance(js, dict):
        for k, v in js.items():
            if k in TOKEN_KEYS and isinstance(v, (str, int)):
                return str(v)
        for v in js.values():
            t = _extract_token_from_json(v)
            if t:
                return t
    elif isinstance(js, list):
        for it in js:
            t = _extract_token_from_json(it)
            if t:
                return t
    return None

_TOKEN_LABELED_RX = re.compile(
    r'(?:(?:secret\s*token|secretToken|token|api[_\s-]*key|apikey|key)\s*[:=]\s*["\']?)([A-Za-z0-9._\-]{16,})',
    re.I
)
_TOKEN_CODE_RX = re.compile(r'<(?:code|pre)[^>]*>\s*([^<>\s]{16,})\s*</(?:code|pre)>', re.I | re.S)
_TOKEN_UNLABELED_HEX_RX = re.compile(r'(?<![A-Za-z0-9])[A-Fa-f0-9]{32,128}(?![A-Za-z0-9])')
_TOKEN_UNLABELED_B64ish_RX = re.compile(r'(?<![A-Za-z0-9])[A-Za-z0-9._\-]{24,256}(?![A-Za-z0-9])')

def _extract_token_from_text(text: str):
    text = (text or "").strip()
    m = _TOKEN_LABELED_RX.search(text)
    if m:
        cand = m.group(1).strip()
        if cand.lower() != "device-width":
            return cand
    m = _TOKEN_CODE_RX.search(text)
    if m:
        cand = m.group(1).strip()
        if cand.lower() != "device-width":
            return cand
    if re.search(r'secret\s*token', text, re.I):
        m = _TOKEN_UNLABELED_HEX_RX.search(text)
        if m:
            return m.group(0)
        for m in _TOKEN_UNLABELED_B64ish_RX.finditer(text):
            cand = m.group(0)
            if cand.lower() != "device-width":
                return cand
    return None

def _handle_mission_url(data: bytes):
    try:
        js = json.loads(data.decode("utf-8", errors="ignore"))
        t = _extract_token_from_json(js)
        if t:
            return t
    except Exception:
        pass
    return _extract_token_from_text(data.decode("utf-8", errors="ignore"))

# ---- City ↔ Landmark (from PDF list) ----
_LANDMARK_CITY_ROWS = [
    ("Gateway of India", "Delhi"),
    ("India Gate", "Mumbai"),
    ("Charminar", "Chennai"),
    ("Marina Beach", "Hyderabad"),
    ("Howrah Bridge", "Ahmedabad"),
    ("Golconda Fort", "Mysuru"),
    ("Qutub Minar", "Kochi"),
    ("Taj Mahal", "Hyderabad"),
    ("Meenakshi Temple", "Pune"),
    ("Lotus Temple", "Nagpur"),
    ("Mysore Palace", "Chandigarh"),
    ("Rock Garden", "Kerala"),
    ("Victoria Memorial", "Bhopal"),
    ("Vidhana Soudha", "Varanasi"),
    ("Sun Temple", "Jaisalmer"),
    ("Golden Temple", "Pune"),
    ("Eiffel Tower", "New York"),
    ("Statue of Liberty", "London"),
    ("Big Ben", "Tokyo"),
    ("Colosseum", "Beijing"),
    ("Sydney Opera House", "London"),
    ("Christ the Redeemer", "Bangkok"),
    ("Burj Khalifa", "Toronto"),
    ("CN Tower", "Dubai"),
    ("Petronas Towers", "Amsterdam"),
    ("Leaning Tower of Pisa", "Cairo"),
    ("Mount Fuji", "San Francisco"),
    ("Niagara Falls", "Berlin"),
    ("Louvre Museum", "Barcelona"),
    ("Stonehenge", "Moscow"),
    ("Sagrada Familia", "Seoul"),
    ("Acropolis", "Cape Town"),
    ("Big Ben", "Istanbul"),
    ("Machu Picchu", "Riyadh"),
    ("Taj Mahal", "Paris"),
    ("Moai Statues", "Dubai Airport"),
    ("Christchurch Cathedral", "Singapore"),
    ("The Shard", "Jakarta"),
    ("Blue Mosque", "Vienna"),
    ("Neuschwanstein Castle", "Kathmandu"),
    ("Buckingham Palace", "Los Angeles"),
    ("Space Needle", "Mumbai"),
    ("Times Square", "Seoul"),
]

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

_CITY_TO_LANDMARK = {}
for lmk, city in _LANDMARK_CITY_ROWS:
    k = _norm(city)
    if k not in _CITY_TO_LANDMARK:
        _CITY_TO_LANDMARK[k] = lmk

def _flight_endpoint_for_landmark(lmk: str) -> str:
    l = _norm(lmk)
    if "gateway of india" in l:
        return "getFirstCityFlightNumber"
    if "taj mahal" in l:
        return "getSecondCityFlightNumber"
    if "eiffel tower" in l:
        return "getThirdCityFlightNumber"
    if "big ben" in l:
        return "getFourthCityFlightNumber"
    return "getFifthCityFlightNumber"

def _get_city_once() -> str:
    city_url = "https://register.hackrx.in/submissions/myFavouriteCity"
    r1 = REQUESTS_SESSION.get(city_url, timeout=12, headers={"Accept": "application/json"})
    r1.raise_for_status()
    try:
        j = r1.json() or {}
        data = j.get("data") if isinstance(j.get("data"), dict) else j
        return (data or {}).get("city", "") or (r1.text or "").strip().strip('"').strip()
    except Exception:
        return (r1.text or "").strip().strip('"').strip()

def _solve_flight_number() -> str:
    cities = []
    for _ in range(3):
        cities.append(_get_city_once())
        time.sleep(0.12)
    city = max(collections.Counter(cities).items(), key=lambda kv: kv[1])[0].strip()
    if not city:
        raise RuntimeError("Favourite city not returned")
    landmark = _CITY_TO_LANDMARK.get(_norm(city), "")
    route = _flight_endpoint_for_landmark(landmark)
    f_url = f"https://register.hackrx.in/teams/public/flights/{route}"
    print(f"[Level-4] City={city} | Landmark={landmark or 'N/A'} | Route={route} | URL={f_url}")
    r2 = REQUESTS_SESSION.get(f_url, timeout=12, headers={"Accept": "application/json"})
    r2.raise_for_status()
    print(f"[Level-4] GET {f_url} → {r2.text[:200]}")
    try:
        j2 = r2.json() or {}
        data2 = j2.get("data") if isinstance(j2.get("data"), dict) else j2
        flight = (data2 or {}).get("flightNumber") or (data2 or {}).get("flight_number") or (data2 or {}).get("flight")
        if not flight:
            m = re.search(r'"?flight[_ ]?number"?\s*[:=]\s*"?([A-Za-z0-9]+)"?', r2.text, flags=re.I)
            flight = m.group(1) if m else ""
    except Exception:
        flight = (r2.text or "").strip().strip('"').strip()
    if not flight:
        raise RuntimeError(f"Flight number missing; raw: {r2.text[:300]}")
    return str(flight).strip()

def _clean_pdf_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t ")
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"[ \t]*\n[ \t]*", "\n", s)
    return s


# ---------------- Routes ----------------



@app.get("/")
def read_root():
    return {"message": "Bajaj Chatbot PDF API is running"}




@app.post("/api/v1/hackrx/run")
async def run_analysis_final(request: RunRequest, authorization: str = Header(...)):
    """
    Final Level-4 route merged into /api/v1/hackrx/run:
    - Handles non-PDF register.hackrx.in URLs (secret token / flight number).
    - >200 pages: tiny OCR snippets + title → WEB prompt (early return; no full extraction).
    - ≤100 pages: PER-QUESTION flow with language enforcement and guard rewrite.
    - 101–200 pages: per-question focused context; targeted retries; language enforced.
    """
    print(f"⏱ Starting run with {len(request.questions)} questions on {request.documents}")
    print(f"Questions: {request.questions}")
    print(f"Documents: {request.documents}")

    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ---- EARLY ESCAPE FOR FLIGHT/TICKET NUMBER ----
    qtext = " ".join(request.questions).lower()
    if re.search(r"\b(flight|ticket)\s*(no\.?|number)\b", qtext):
        try:
            flight = _solve_flight_number()
            return {"answers": [flight]}
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Flight solver failed: {e}")
    # ------------------------------------------------

    start = time.time()
    try:
        # Fetch once
        r = REQUESTS_SESSION.get(request.documents, timeout=20)
        r.raise_for_status()
        data = r.content
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].lower().strip()

        # Detect PDF vs mission URL
        is_pdf = _is_pdf_payload(data, ctype)
        if not is_pdf and _is_mission_host(request.documents):
            if re.search(r"\b(flight|ticket)\s*(no\.?|number)\b", qtext):
                try:
                    return {"answers": [_solve_flight_number()]}
                except Exception as e:
                    raise HTTPException(status_code=502, detail=f"Flight solver failed: {e}")
            if any(x in qtext for x in ("token", "secret", "api key", "apikey", "key")):
                token = _handle_mission_url(data)
                return {"answers": [token] if token else ["Not found in non-PDF URL."]}
            return {"answers": ["Not found in non-PDF URL."]}

        # ---------- Ultra-fast tiny OCR + meta for big PDFs ----------
        page_count_fast, title_fast, tiny_snippets = _fast_title_and_snippets_with_ocr(data)

        # >200 pages → hybrid path: tiny OCR context + public/web knowledge
        if page_count_fast and page_count_fast > 200:
            try:
                qblock = make_question_block(request.questions)
                hybrid_title = (title_fast or "Untitled Document")
                if tiny_snippets:
                    hybrid_title = f"{hybrid_title}\n\nKey OCR snippets:\n{tiny_snippets}"
                m_params = choose_mistral_params(page_count_fast, hybrid_title)
                m_params["temperature"] = min(m_params.get("temperature", 0.2), 0.18)
                resp = call_mistral(
                    WEB_PROMPT_TEMPLATE.format(title=hybrid_title, query=qblock),
                    m_params
                )
                cleaned = [_sanitize_line(ln) for ln in resp.splitlines() if ln.strip()]
                answers = cleaned[:len(request.questions)] if cleaned else ["Not found in public sources."] * len(request.questions)
                return {"answers": answers}
            except Exception:
                return {"answers": ["Not found in public sources."] * len(request.questions)}
        # ----------------------------------------------------------------

        # ≤200: Extract full text
        full_text, page_count, _title = extract_text_from_pdf_url(request.documents)
        full_text = _clean_pdf_text(full_text)
        if not full_text:
            raise HTTPException(status_code=422, detail="Could not extract text from PDF.")
        chunks = split_text(full_text) if full_text else []

        # ≤100 pages: PER-QUESTION flow (language enforced + guard rewrite)
        if page_count <= 100:
            answers: List[str] = []
            for q in request.questions:
                try:
                    idxs = _topk_indices(q, chunks, k=6)
                    ctx = _context_with_neighbors(chunks, idxs, neighbor=1, budget_chars=9500)
                    m_params = choose_mistral_params(page_count, ctx)
                    lang_rule = enforce_lang_instruction(q)
                    sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."
                    prompt = CHUNK_PROMPT_TEMPLATE.format(context=ctx, query=q, lang_rule=lang_rule)
                    a = _sanitize_line(call_mistral(prompt, m_params, system=sys_msg))
                    if not a or _is_not_found(a):
                        a = await _retry_per_question(q, full_text, chunks, page_count)
                except Exception:
                    a = await _retry_per_question(q, full_text, chunks, page_count)

                # Final language guard: quick rewrite if the model slips
                try:
                    target_lang = question_lang_label(q)
                    if target_lang == "Malayalam" and not is_malayalam(a):
                        a = _sanitize_line(call_mistral(
                            f"Rewrite the following answer in Malayalam without adding new facts:\n\n{a}",
                            {"max_tokens": 300, "temperature": 0.0, "timeout": 10},
                            system="Output only the rewritten answer text."
                        ))
                    elif target_lang == "English" and is_malayalam(a):
                        a = _sanitize_line(call_mistral(
                            f"Rewrite the following answer in English without adding new facts:\n\n{a}",
                            {"max_tokens": 300, "temperature": 0.0, "timeout": 10},
                            system="Output only the rewritten answer text."
                        ))
                except Exception:
                    pass

                # >>> Postshape for clean, leaderboard-friendly formatting <<<
                a = _postshape(a, question_lang_label(q))
                answers.append(a)

            return {"answers": answers[:len(request.questions)]}

        # 101–200 pages: per-question focus + retry (language enforced)
        elif page_count <= 200:
            out: List[str] = []
            for q in request.questions:
                try:
                    idxs = _topk_indices(q, chunks, k=6)
                    ctx = _context_with_neighbors(chunks, idxs, neighbor=1, budget_chars=9500)
                    m_params = choose_mistral_params(page_count, ctx)
                    lang_rule = enforce_lang_instruction(q)
                    sys_msg = "Output only the final answer text as one short paragraph. No labels, numbering, bullets, or extra commentary. Answer strictly in the same language as the question."
                    a = _sanitize_line(call_mistral(
                        CHUNK_PROMPT_TEMPLATE.format(context=ctx, web_snippets="", query=q, lang_rule=lang_rule),
                        m_params,
                        system=sys_msg
                    ))
                    if not a or _is_not_found(a):
                        a = await _retry_per_question(q, full_text, chunks, page_count)

                    # >>> Postshape before appending <<<
                    a = _postshape(a, question_lang_label(q))
                    out.append(a)

                except Exception:
                    # >>> Postshape even on retry path <<<
                    tmp = await _retry_per_question(q, full_text, chunks, page_count)
                    out.append(_postshape(tmp, question_lang_label(q)))

            return {"answers": out}

        return {"answers": ["Not found in public sources."] * len(request.questions)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    finally:
        print(f"⏱ Final total time: {round(time.time() - start, 2)}s")

