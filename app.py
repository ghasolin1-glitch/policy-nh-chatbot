# app.py — 보험 약관 RAG 챗봇 (NHLife 스타일 UI, GPT-5 + Supabase pgvector)
import os
import json
import time
import typing as t
import numpy as np
import psycopg
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

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
model = ChatOpenAI(model='gpt-5', reasoning_effort='minimal', api_key=OPENAI_API_KEY)

# =========================
# 🧱 Streamlit UI 설정
# =========================
st.set_page_config(page_title="약관챗봇", page_icon="📘", layout="centered")

st.markdown("""
<style>
    /* 전체 배경 */
    body { background-color: #f3f4f6; }

    /* 헤더 */
    .chat-header {
        background-color: #2563eb;
        color: white;
        padding: 16px;
        border-radius: 10px 10px 0 0;
        text-align: left;
        font-family: Pretendard, sans-serif;
    }
    .chat-header h1 {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 0;
    }
    .chat-header p {
        font-size: 0.8rem;
        color: #bfdbfe;
        margin: 0;
    }

    /* 채팅 영역 */
    .chat-box {
        background-color: white;
        height: 550px;
        overflow-y: auto;
        padding: 16px;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* 말풍선 */
    .bubble {
        padding: 10px 14px;
        border-radius: 20px;
        margin-bottom: 8px;
        max-width: 80%;
        display: inline-block;
        word-wrap: break-word;
        line-height: 1.5;
    }

    /* 사용자 말풍선 (오른쪽) */
    .user-bubble {
        background-color: #2563eb;
        color: white;
        border-bottom-right-radius: 4px;
        float: right;
        clear: both;
    }

    /* 챗봇 말풍선 (왼쪽) */
    .bot-bubble {
        background-color: #e5e7eb;
        color: #111827;
        border-bottom-left-radius: 4px;
        float: left;
        clear: both;
    }

    .timestamp {
        font-size: 0.7rem;
        color: #9ca3af;
        margin-top: 2px;
    }

    /* 입력창 */
    .input-box {
        display: flex;
        margin-top: 10px;
        gap: 8px;
    }
    .input-box input {
        flex: 1;
        padding: 10px 16px;
        border-radius: 999px;
        border: 1px solid #d1d5db;
        outline: none;
    }
    .input-box button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 50%;
        width: 44px;
        height: 44px;
        font-size: 1.2rem;
        cursor: pointer;
    }
    .input-box button:hover {
        background-color: #1d4ed8;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# 📦 세션 상태 관리
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}
    ]

# =========================
# 📦 DB 연결 및 검색 함수
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
        return f"오류가 발생했습니다: {e}"

# =========================
# 🚀 UI 출력
# =========================
st.markdown('<div class="chat-header"><h1>약관챗봇</h1><p>NHLife | Made by 태훈,현철</p></div>', unsafe_allow_html=True)
st.markdown('<div class="chat-box">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='bubble user-bubble'>{msg['content']}</div><div class='timestamp' style='text-align:right'>{time.strftime('%H:%M')}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble bot-bubble'>{msg['content']}</div><div class='timestamp'>{time.strftime('%H:%M')}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ✉️ 입력창 (하단)
# =========================
with st.form("chat_input", clear_on_submit=True):
    st.markdown('<div class="input-box">', unsafe_allow_html=True)
    user_input = st.text_input("", placeholder="상품에 대해 궁금한 점 질문해주세요.", label_visibility="collapsed")
    submit = st.form_submit_button("📤")
    st.markdown("</div>", unsafe_allow_html=True)

if submit and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = generate_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
