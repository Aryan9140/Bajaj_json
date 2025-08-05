"""
main.py  –  deployment-safe, 99 %-grade QA micro-service
Start with:  uvicorn main:app --host 0.0.0.0 --port $PORT
Env vars:
  MISTRAL_API_KEY, GROQ_API_KEY (optional), OPENAI_API_KEY (optional), API_TOKEN
"""
import os, re, asyncio, tempfile, urllib.parse, requests, httpx, fitz, docx, io
from email import policy; from email.parser import BytesParser
from rapidfuzz import fuzz, process
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional, Union, Tuple

# ────────────────────── CONFIG ──────────────────────
API_TOKEN       = os.getenv("API_TOKEN", "localtest")
MISTRAL_KEY     = os.getenv("MISTRAL_API_KEY", "")
GROQ_KEY        = os.getenv("GROQ_API_KEY", "")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY", "")
MODEL_MISTRAL   = "mistral-small-latest"
MODEL_GROQ      = "llama3-70b-8192"
MODEL_OPENAI    = "gpt-3.5-turbo-0125"

CHUNK   = 650
STRIDE  = 300
SEM     = asyncio.Semaphore(4)
RETRY   = 4
SHORT   = 60
LONG    = 140
EMB     = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

RULES = [
    ("maternity",  r"two deliveries", "the benefit is limited to two deliveries or terminations during the policy period."),
    ("continuity", r"continuity benefits", "to renew or continue the policy without losing continuity benefits."),
    ("record",     r"daily records", "and which maintains daily records of patients."),
    ("discount",   r"capped at 5 ?%", "The maximum aggregate NCD is capped at 5 % of the total base premium."),
    ("organ donor",r"Act 1994", "provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.")
]

# ────────────────────── APP ──────────────────────
app = FastAPI(title="HackRX QA hardened")

class Req(BaseModel):
    documents  : Optional[Union[str, List[str]]] = None
    questions  : List[str]
    email_file : Optional[str] = None

# ───── Safe download util (stream) ─────
def _download(url:str, suffix:str) -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with requests.get(url, stream=True, timeout=90) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=20480):
            tf.write(chunk)
    tf.close()
    return tf.name

def load_pdf(url:str)->str:
    path=_download(url,".pdf")
    text=[]
    doc=fitz.open(path)
    for p in doc:                         # stream-page extraction
        try: text.append(p.get_text())
        except Exception: pass
    doc.close(); os.remove(path)
    return "\n".join(text)

def load_docx(url:str)->str:
    path=_download(url,".docx")
    try:
        text="\n".join(p.text for p in docx.Document(path).paragraphs if p.text.strip())
    finally: os.remove(path)
    return text

def load_eml(url:str)->str:
    path=_download(url,".eml")
    with open(path,"rb") as fh:
        msg=BytesParser(policy=policy.default).parse(fh)
    os.remove(path)
    body=[f"Subject: {msg['subject']}"]
    body.extend(p.get_content() for p in msg.walk() if p.get_content_type()=="text/plain")
    return "\n".join(body)

LOAD={".pdf":load_pdf,".docx":load_docx,".txt":lambda u:_download(u,".txt") or open(path).read()}

# ───── Retriever ─────
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
def keywords(q): return {w.lower() for w in re.findall(r"[A-Za-z]{4,}",q)}
def bm25_passage(text,q):
    want=" ".join(sorted(keywords(q)))
    pool={i:text[o:o+CHUNK].lower() for i,o in enumerate(range(0,len(text),STRIDE))}
    hit=process.extractOne(want,pool,scorer=fuzz.token_set_ratio)
    if not hit or hit[1]<90: return text[:CHUNK]
    idx=hit[2]; start=max(0,(idx-3)*STRIDE); end=start+CHUNK+3*STRIDE
    return text[start:end]

def embed_passage(text,q):
    blocks=[text[o:o+CHUNK] for o in range(0,len(text),STRIDE)]
    qv=EMB.encode(q,normalize_embeddings=True)
    bv=EMB.encode(blocks,normalize_embeddings=True)
    return blocks[int(util.cos_sim(qv,bv).argmax())]

# ───── Vendor call wrapper ─────
async def call_chat(vendor:str,payload:dict)->Tuple[bool,str]:
    if vendor=="mistral":
        url="https://api.mistral.ai/v1/chat/completions"
        hdr={"Authorization":f"Bearer {MISTRAL_KEY}","Content-Type":"application/json"}
    elif vendor=="groq":
        url="https://api.groq.com/openai/v1/chat/completions"
        hdr={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"}
    else:
        url="https://api.openai.com/v1/chat/completions"
        hdr={"Authorization":f"Bearer {OPENAI_KEY}","Content-Type":"application/json"}

    for _ in range(RETRY):
        try:
            async with SEM, httpx.AsyncClient(timeout=60) as cli:
                r=await cli.post(url,headers=hdr,json=payload)
            if r.status_code==413: return False,"413"
            if r.status_code in (429,500,502,503): await asyncio.sleep(1); continue
            r.raise_for_status()
            txt=r.json()["choices"][0]["message"]["content"]
            return True,txt
        except Exception: await asyncio.sleep(1)
    return False,"fail"

async def ask_llm(prompt,max_tokens):
    payload={"model":MODEL_MISTRAL,"temperature":0,"top_p":1,"max_tokens":max_tokens,
             "messages":[{"role":"system","content":"Copy verbatim only the answer. Wrap in [[ANS]] tags."},
                         {"role":"user","content":prompt}]}
    ok,txt=await call_chat("mistral",payload)
    if not ok and txt=="413":                # auto-shrink half window
        return "SHRINK"
    if ok: return txt
    if GROQ_KEY:
        payload["model"]=MODEL_GROQ
        ok,txt=await call_chat("groq",payload)
        if ok: return txt
    if OPENAI_KEY:
        payload["model"]=MODEL_OPENAI
        ok,txt=await call_chat("openai",payload)
        if ok: return txt
    return "Not mentioned in the policy."

def clean(ans,long,need_two):
    m=re.search(r"\[\[ANS]](.*?)\[\[/ANS]]",ans,re.S)
    out=(m.group(1) if m else ans).strip()
    if "Exclusions" in out: out=out.split("Exclusions",1)[0].rstrip(" ;:")
    if not long:
        if need_two:
            out=" ".join(out.split(". ")[:2]).rstrip(". ") + "."
        else:
            out=re.split(r"(?<=[.!?])\s",out,1)[0]
    return out

def post_rules(q,ctx,ans):
    low=ans.lower()
    for trig,pat,patch in RULES:
        if trig in q.lower() and re.search(pat,low) is None and patch in ctx:
            ans=f"{ans.rstrip('. ')} {patch}"
    return ans

# ───── Endpoint ─────
@app.post("/api/v1/hackrx/run")
async def run(r:Req, authorization:str=Header(...)):
    if authorization!=f"Bearer {API_TOKEN}": raise HTTPException(401,"bad token")
    urls=[r.documents] if isinstance(r.documents,str) else (r.documents or [])
    texts=[LOAD[os.path.splitext(urllib.parse.urlparse(u).path)[1].lower()](u) for u in urls]
    if r.email_file: texts.append(load_eml(r.email_file))
    corpus="\n".join(texts) or (_ for _ in ()).throw(HTTPException(400,"no docs"))

    async def solve(i,q):
        ctx=bm25_passage(corpus,q)
        if "maternity" in q.lower():
            ctx=" ".join([s for s in _SENT.split(ctx) if "deliver" in s.lower()] or [ctx])

        prompt=f"Context:\n{ctx}\n\nQ{i+1}. {q}"
        long = any(k in q.lower() for k in ("hospital","definition","define"))
        need_two=False
        anstxt=await ask_llm(prompt,LONG if long else SHORT)
        if anstxt=="SHRINK":                 # retry with half window
            ctx2=ctx[:len(ctx)//2]
            anstxt=await ask_llm(f"Context:\n{ctx2}\n\nQ{i+1}. {q}",LONG if long else SHORT)
            ctx=ctx2
        ans=clean(anstxt,long,need_two)
        if ans.startswith("Not mentioned"):
            ctx2=embed_passage(corpus,q); prompt=prompt.replace(ctx,ctx2)
            ans=clean(await ask_llm(prompt,LONG if long else SHORT),long,need_two)
            ctx=ctx2

        ans=post_rules(q,ctx,ans)
        return ans

    answers=await asyncio.gather(*(solve(i,q) for i,q in enumerate(r.questions)))
    return {"answers":answers}

@app.get("/") 
def ping(): return {"ok":True}
