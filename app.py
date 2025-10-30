# app.py — 보험 약관 RAG 챗봇 (HTML UI + GPT-5 + Supabase pgvector)
import os, json, time, typing as t, numpy as np, psycopg, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import streamlit.components.v1 as components

# =========================
# 🔧 환경 변수
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")

missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "DB_HOST": DB_HOST, "DB_PORT": DB_PORT,
    "DB_NAME": DB_NAME, "DB_USER": DB_USER, "DB_PASS": DB_PASS
}.items() if not v]
if missing:
    st.error(f"환경 변수 누락: {', '.join(missing)}")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
model = ChatOpenAI(model="gpt-5", reasoning_effort="minimal", api_key=OPENAI_API_KEY)

# =========================
# 📦 세션 상태
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# =========================
# 📦 DB 설정
# =========================
DB_CONN = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASS,
    "sslmode": "require",
    "connect_timeout": 10,
}

SEARCH_SQL = """
WITH q AS (SELECT %(vec)s::vector AS v)
SELECT
  pdf_filename,
  COALESCE((metadata_json->>'pdf_path'), pdf_path) AS pdf_path,
  page,
  text,
  (1 - (embedding <=> (SELECT v FROM q))) AS cosine_similarity
FROM public.terms_chunks
ORDER BY cosine_similarity DESC
LIMIT %(k)s;
"""

def to_vector_literal(vec):
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def embed_text(text: str) -> list[float]:
    e = client.embeddings.create(model="text-embedding-3-small", input=text)
    v = np.array(e.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else v.tolist()

def fetch_topk_chunks(q_vec, k=5):
    vec_literal = to_vector_literal(q_vec)
    with psycopg.connect(**DB_CONN) as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, {"vec": vec_literal, "k": k})
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

# =========================
# 💬 답변 함수
# =========================
def generate_answer(question: str) -> str:
    try:
        q_vec = embed_text(question)
        df = fetch_topk_chunks(q_vec, 5)
        if df.empty:
            return "죄송하지만 관련 약관 내용을 찾지 못했습니다."

        context = "\n\n".join(df["text"].head(3))
        sys_prompt = f"""
        당신은 대한민국 보험 약관 전문 챗봇입니다.
        아래 약관 내용을 참고하여 사용자의 질문에 근거 기반으로 답변하세요.

        [약관 관련 조항]
        {context}
        """
        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=question)]
        resp = model.invoke(msgs)
        return resp.content
    except Exception as e:
        return f"⚠️ 오류가 발생했습니다: {e}"

# =========================
# 🖥️ HTML UI 렌더링
# =========================
chat_body_html = ''.join([
    f"<div class='{'user-bubble' if m['role']=='user' else 'bot-bubble'} bubble'>{m['content']}</div>"
    for m in st.session_state.messages
])

html_code = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<script src="https://cdn.tailwindcss.com"></script>
<style>
body {{
  background-color: #f3f4f6;
  font-family: 'Pretendard', 'Inter', sans-serif;
}}
.chat-container {{
  width: 100%;
  max-width: 480px;
  margin: 0 auto;
  background: white;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  display: flex;
  flex-direction: column;
  height: 85vh;
  overflow: hidden;
}}
.chat-header {{
  background-color: #2563eb;
  color: white;
  padding: 16px;
}}
.chat-header h1 {{
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0;
}}
.chat-header p {{
  font-size: 0.8rem;
  color: #bfdbfe;
  margin: 0;
}}
.chat-body {{
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}}
.bubble {{
  padding: 10px 14px;
  border-radius: 20px;
  margin-bottom: 10px;
  max-width: 80%;
  line-height: 1.5;
  word-wrap: break-word;
}}
.user-bubble {{
  background-color: #2563eb;
  color: white;
  border-bottom-right-radius: 4px;
  margin-left: auto;
}}
.bot-bubble {{
  background-color: #e5e7eb;
  color: #111827;
  border-bottom-left-radius: 4px;
  margin-right: auto;
}}
.chat-input {{
  border-top: 1px solid #e5e7eb;
  padding: 12px;
  display: flex;
  gap: 8px;
}}
.chat-input input {{
  flex: 1;
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid #d1d5db;
}}
.chat-input button {{
  background-color: #2563eb;
  color: white;
  border: none;
  border-radius: 50%;
  width: 44px;
  height: 44px;
  font-size: 1.2rem;
  cursor: pointer;
}}
</style>
</head>
<body>
<div class="chat-container">
  <div class="chat-header">
    <h1>약관챗봇</h1>
    <p>NHLife | Made by 태훈,현철</p>
  </div>
  <div class="chat-body" id="chat-body">{chat_body_html}</div>
</div>
</body>
</html>
"""

components.html(html_code, height=800, scrolling=False)

# =========================
# ✏️ 입력창 (Streamlit 폼)
# =========================
with st.form("chat_input", clear_on_submit=True):
    user_input = st.text_input("", placeholder="상품에 대해 궁금한 점 질문해주세요.", label_visibility="collapsed")
    submitted = st.form_submit_button("📤")

if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = generate_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
