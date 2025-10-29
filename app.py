# app.py — 보험 약관 RAG 챗봇 (Cosine Similarity, Supabase pgvector, GPT-5)
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
# 🧱 Streamlit UI
# =========================
st.set_page_config(page_title="약관 RAG 챗봇", page_icon="🤖", layout="wide")
st.markdown("""
<style>
  .glow-input input { box-shadow: 0 0 12px rgba(0,120,255,0.45); border-radius: 999px; }
  .small { font-size: 0.9rem; color: #666; }
  .cite { font-size: 0.9rem; color: #3b82f6; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.markdown("## 🤖 보험 약관 RAG 챗봇")
    st.caption("Supabase(pgvector)에서 유사 청크 검색 → GPT-5 답변 (Cosine Similarity)")
with col2:
    with st.popover("⚙️ 옵션"):
        top_k = st.slider("검색 Top-K", 1, 20, 10, 1)
        max_ctx_chars = st.slider("컨텍스트 최대 길이(문자)", 2000, 12000, 6000, 500)
        temperature = st.slider("창의성(Temperature)", 0.0, 1.2, 0.2, 0.1)
        sys_style = st.selectbox("응답 스타일",
                                 ["간결 요약", "근거 중심", "친절 설명"],
                                 index=1)

st.markdown('<div class="glow-input">', unsafe_allow_html=True)
question = st.text_input("질문을 입력하세요 (예: 369뉴테크NH암보험 암수술자금은 얼마인가요?)", "")
st.markdown("</div>", unsafe_allow_html=True)
ask = st.button("실행", type="primary", use_container_width=True)

# =========================
# 🗄️ DB 연결 및 Cosine 검색 SQL
# =========================
DB_CONN = {
    "host": DB_HOST,
    "port": DB_PORT,
    "dbname": DB_NAME,
    "user": DB_USER,
    "password": DB_PASS,
    "sslmode": "require",
    "connect_timeout": 10,
    "options": "-c prepare_threshold=0 -c tcp_keepalives_idle=20 "
               "-c tcp_keepalives_interval=10 -c tcp_keepalives_count=3",
}

SEARCH_SQL = """
WITH q AS (SELECT %(vec)s::vector AS v)
SELECT
  pdf_filename,
  COALESCE((metadata_json->>'pdf_path'), pdf_path) AS pdf_path,
  page,
  chunk_id,
  text,
  metadata_json,
  (1 - (embedding <=> (SELECT v FROM q))) AS cosine_similarity
FROM public.terms_chunks
ORDER BY cosine_similarity DESC
LIMIT %(k)s;
"""

def to_vector_literal(vec: t.List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def embed_text(text: str, model_name: str = "text-embedding-3-small") -> t.List[float]:
    e = client.embeddings.create(model=model_name, input=text)
    v = np.array(e.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(v)
    return (v / norm).tolist() if norm > 0 else v.tolist()

def fetch_topk_chunks(q_vec: t.List[float], k: int) -> pd.DataFrame:
    vec_literal = to_vector_literal(q_vec)
    with psycopg.connect(**DB_CONN) as conn, conn.cursor() as cur:
        cur.execute(SEARCH_SQL, {"vec": vec_literal, "k": k})
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

# =========================
# 📦 VectorStore 어댑터 (DB 기반)
# =========================
class DBStore:
    """LangChain VectorStore처럼 동작하는 어댑터."""
    def __init__(self, default_k: int = 10):
        self.default_k = default_k
        self.last_df: pd.DataFrame | None = None

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        k = k or self.default_k
        q_vec = embed_text(query)
        df = fetch_topk_chunks(q_vec, k)
        self.last_df = df
        docs: list[Document] = []
        for _, r in df.iterrows():
            meta = {
                "pdf_filename": r["pdf_filename"],
                "pdf_path": r.get("pdf_path"),
                "page": int(r.get("page") or 0),
                "chunk_id": r.get("chunk_id"),
                "cosine_similarity": float(r.get("cosine_similarity") or 0.0),
                "metadata_json": r.get("metadata_json"),
            }
            docs.append(Document(page_content=str(r["text"] or "").strip(), metadata=meta))
        return docs

store = DBStore(default_k=10)

# =========================
# 💬 질의 함수 (Top 3 컨텍스트 버전)
# =========================
def query(question: str) -> str:
    store.default_k = top_k  # UI 값 반영
    results = store.similarity_search(question)
    if not results:
        return "유사한 청크를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."

    # ✅ 상위 3개의 청크를 모두 컨텍스트로 결합
    top_n = min(3, len(results))
    combined_context = "\n\n---\n\n".join(
        f"[{i+1}] {r.page_content}" for i, r in enumerate(results[:top_n])
    )

    system = f"""
    당신은 대한민국 보험사의 공식 약관을 이해하고 설명하는 전문 AI 상담사입니다.
    아래는 보험약관에서 추출된 Top {top_n}개의 관련 조항입니다.
    각 조항을 종합하여 사용자의 질문에 대해 **정확하고 신뢰성 있는 답변**을 작성하세요.

    [지침]
    1. 답변은 반드시 아래 약관 내용(컨텍스트)에 근거해야 합니다.
    2. 근거가 명확히 존재하지 않으면 “약관에서 해당 내용은 명시되어 있지 않습니다.”라고 답하세요.
    3. 모호하거나 중복된 표현은 정리하고, 핵심만 간결하게 요약하세요.
    4. 숫자(보험금, 지급한도, 기간 등)는 약관 내 표기대로 유지하세요.
    5. 사용자가 실제 보험계약자라고 가정하고, 이해하기 쉽게 자연스럽게 설명하세요.
    6. 법률적, 계약적 표현은 존칭체로 답변하세요. (예: “지급됩니다.”, “해당되지 않습니다.”)
    7. 끝에 간단히 요약 문장을 덧붙이세요. (예: “요약하면, 암수술자금은 진단 후 1회 지급됩니다.”)

    [약관 관련 조항 요약]
    {combined_context}
    """.strip()

    messages = [
        SystemMessage(system),
        HumanMessage(question)
    ]

    resp = model.invoke(messages)
    return resp.content

# =========================
# 📑 근거 표시
# =========================
def render_citations(df: pd.DataFrame):
    if df.empty:
        return
    st.markdown("**참고한 근거 (Top-K):**")
    for i, r in df.iterrows():
        st.markdown(
            f"- <span class='cite'>[{i+1}] {r['pdf_filename']} p.{int(r['page'])}</span> "
            f"<span class='small'>(유사도: {r['cosine_similarity']:.4f})</span>",
            unsafe_allow_html=True
        )

# =========================
# 🚀 실행부
# =========================
if ask:
    if not question.strip():
        st.warning("질문을 입력해주세요.")
        st.stop()

    with st.spinner("검색 및 GPT-5 답변 생성 중..."):
        try:
            answer = query(question)
        except Exception as e:
            st.error(f"오류: {e}")
            st.stop()

    st.markdown("### 📌 답변")
    st.write(answer)
    st.divider()

    if getattr(store, "last_df", None) is not None and not store.last_df.empty:
        render_citations(store.last_df)
    else:
        st.caption("근거를 표시할 검색 결과가 없습니다.")
else:
    st.info("좌측 입력창에 질문을 쓰고 **실행**을 눌러주세요.")
