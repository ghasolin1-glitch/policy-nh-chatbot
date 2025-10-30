import os, json, time, numpy as np, psycopg, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

client = OpenAI(api_key=OPENAI_API_KEY)
model = ChatOpenAI(model="gpt-5", reasoning_effort="minimal", api_key=OPENAI_API_KEY)

# =========================
# 📦 세션 상태
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"}]

# =========================
# 📦 DB 연결 함수
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
SELECT pdf_filename, COALESCE((metadata_json->>'pdf_path'), pdf_path) AS pdf_path, 
       page, text, (1 - (embedding <=> (SELECT v FROM q))) AS cosine_similarity
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
# 🎨 UI 구성 (Streamlit)
# =========================
st.markdown("""
<style>
body { background-color: #f3f4f6; font-family: Pretendard, Inter, sans-serif; }
.chat-box {
    background: white; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    width: 100%; max-width: 480px; margin: auto; padding: 16px;
}
.chat-header { background-color: #2563eb; color: white; padding: 16px; border-radius: 8px; margin-bottom: 12px; }
.user-bubble {
    background-color: #2563eb; color: white; border-radius: 20px; padding: 10px 14px;
    margin-bottom: 8px; margin-left: auto; max-width: 80%;
}
.bot-bubble {
    background-color: #e5e7eb; color: #111827; border-radius: 20px; padding: 10px 14px;
    margin-bottom: 8px; margin-right: auto; max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-header'><h3>약관챗봇</h3><p>NHLife | Made by 태훈,현철</p></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="상품에 대해 궁금한 점 질문해주세요.", label_visibility="collapsed")
        submitted = st.form_submit_button("📤")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# 💬 입력 처리
# =========================
if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    answer = generate_answer(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
