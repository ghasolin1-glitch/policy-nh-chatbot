# app.py — Streamlit RAG Chatbot (Supabase pgvector 검색, 저장 로직 없음)
import os
import json
import time
import typing as t
import psycopg
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()





# =========================
# 🔧 환경변수 / secrets 읽기
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
DB_HOST = os.getenv("DB_HOST") or st.secrets.get("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT") or st.secrets.get("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME") or st.secrets.get("DB_NAME")
DB_USER = os.getenv("DB_USER") or st.secrets.get("DB_USER")
DB_PASS = os.getenv("DB_PASS") or st.secrets.get("DB_PASS")



# 필수 값 체크
missing = [k for k, v in {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "DB_HOST": DB_HOST, "DB_PORT": DB_PORT, "DB_NAME": DB_NAME, "DB_USER": DB_USER, "DB_PASS": DB_PASS
}.items() if not v]
if missing:
    st.error(f"환경변수/시크릿 누락: {', '.join(missing)}")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# 전역 모델 객체 생성 (Streamlit 실행 시 1회만)
model = ChatOpenAI(
    model="gpt-5",
    reasoning_effort="minimal",
    api_key=OPENAI_API_KEY
)


# =========================
# 🧱 Streamlit UI
# =========================
st.set_page_config(page_title="약관 RAG 챗봇", page_icon="🤖", layout="wide")
st.markdown(
    """
    <style>
      .glow-input input { box-shadow: 0 0 12px rgba(0,120,255,0.45); border-radius: 999px; }
      .small { font-size: 0.9rem; color: #666; }
      .cite { font-size: 0.9rem; color: #3b82f6; }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("## 🤖 보험 약관 RAG 챗봇")
    st.caption("Supabase(pgvector)에서 유사 청크 검색 → GPT-5 답변 (저장/인서트 없음)")

with col2:
    with st.popover("⚙️ 옵션"):
        top_k = st.slider("검색 Top-K", 1, 8, 4, 1)
        max_ctx_chars = st.slider("컨텍스트 최대 길이(문자)", 2000, 12000, 6000, 500)
        temperature = st.slider("창의성(Temperature)", 0.0, 1.2, 0.2, 0.1)
        sys_style = st.selectbox(
            "응답 스타일",
            ["간결 요약", "근거 중심", "친절 설명"],
            index=1
        )

st.markdown('<div class="glow-input">', unsafe_allow_html=True)
question = st.text_input("질문을 입력하세요 (예: 369뉴테크NH암보험 암수술자금은 얼마인가요?)", "")
st.markdown("</div>", unsafe_allow_html=True)

ask = st.button("실행", type="primary", use_container_width=True)

# =========================
# 🗄️ DB 연결 준비
# =========================
DB_CONN: dict = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASS,
    "sslmode": "require",
    "connect_timeout": 10,
    # pgbouncer(풀러) 연결 안정화 옵션
    "options": "-c prepare_threshold=0 -c tcp_keepalives_idle=20 -c tcp_keepalives_interval=10 -c tcp_keepalives_count=3",
}

SEARCH_SQL = """
WITH q AS (
  SELECT {vec}::vector AS v
)
SELECT
  pdf_filename,
  COALESCE((metadata_json->>'pdf_path'), pdf_path) AS pdf_path,
  page,
  chunk_id,
  text,
  metadata_json,
  (embedding <=> (SELECT v FROM q)) AS cosine_distance
FROM public.terms_chunks
ORDER BY cosine_distance ASC
LIMIT %(k)s;
"""

def to_vector_literal(vec: t.List[float]) -> str:
    # pgvector 입력 포맷: [0.1, 0.2, ...]
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def embed_text(text: str, model: str = "text-embedding-3-small") -> t.List[float]:
    e = client.embeddings.create(model=model, input=text)
    return e.data[0].embedding

def fetch_topk_chunks(q_vec: t.List[float], k: int) -> pd.DataFrame:
    vec_literal = to_vector_literal(q_vec)
    # SQL 안에서 vec_literal을 문자열 포맷으로 직접 삽입
    sql_stmt = SEARCH_SQL.format(vec=f"'{vec_literal}'")

    with psycopg.connect(**DB_CONN) as conn, conn.cursor() as cur:
        cur.execute(sql_stmt, {"k": k})
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def build_context(df: pd.DataFrame, max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for _, r in df.iterrows():
        header = f"[{r['pdf_filename']} p.{int(r['page'])}] {r['chunk_id']}"
        chunk = str(r["text"]).strip()
        block = f"### {header}\n{chunk}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)

def system_prompt(style: str) -> str:
    base = (
        "당신은 보험 약관 전문 요약/설명 도우미입니다. 아래 컨텍스트를 근거로만 답하세요.\n"
        "- 근거 없는 추측/상상 금지, 숫자는 원문 기준으로 정확히.\n"
        "- 출처(파일명·페이지)를 함께 표기하세요.\n"
    )
    if style == "간결 요약":
        base += "- 한글로 3~6문장, 핵심만 간결하게 정리.\n"
    elif style == "근거 중심":
        base += "- 답변 뒤에 관련 조항/문구를 짧게 인용(따옴표)하고 출처를 명시.\n"
    elif style == "친절 설명":
        base += "- 배경과 조건을 쉽고 친절하게 풀어 설명.\n"
    return base

def chat_answer(question: str, context: str) -> str:
    """원래 코드와 동일한 구조의 GPT 호출"""
    system_prompt = f"아래 내용을 바탕으로 사용자의 질문에 대답해줘.\n\n{context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]

    resp = model.invoke(messages)
    return resp.content


def render_citations(df: pd.DataFrame):
    if df.empty:
        return
    st.markdown("**참고한 근거 (Top-K):**")
    for i, r in df.iterrows():
        sim = 1 - float(r["cosine_distance"])  # 유사도(1 - distance)
        st.markdown(
            f"- <span class='cite'>[{i+1}] {r['pdf_filename']} p.{int(r['page'])}</span> "
            f"<span class='small'>(유사도: {sim:.4f})</span>",
            unsafe_allow_html=True
        )

# =========================
# 🚀 실행
# =========================
if ask:
    if not question.strip():
        st.warning("질문을 입력해주세요.")
        st.stop()

    with st.spinner("질문 임베딩 생성 중..."):
        q_vec = embed_text(question)

    with st.spinner("Supabase에서 유사 청크 검색 중..."):
        try:
            df_hits = fetch_topk_chunks(q_vec, top_k)
        except Exception as e:
            st.error(f"검색 중 오류: {e}")
            st.stop()

    if df_hits.empty:
        st.info("유사한 청크를 찾지 못했습니다. 질문을 더 구체적으로 바꿔보세요.")
        st.stop()

    ctx = build_context(df_hits, max_chars=max_ctx_chars)

    with st.spinner("답변 생성 중..."):
        try:
            answer = chat_answer(question, ctx)
        except Exception as e:
            st.error(f"모델 호출 오류: {e}")
            st.stop()

    st.markdown("### 📌 답변")
    st.write(answer)
    st.divider()
    render_citations(df_hits)

    with st.expander("🔎 검색 컨텍스트 미리보기"):
        st.markdown(f"<div class='mono' style='white-space:pre-wrap'>{ctx}</div>", unsafe_allow_html=True)

else:
    st.info("좌측 입력창에 질문을 쓰고 **실행**을 눌러주세요.")
