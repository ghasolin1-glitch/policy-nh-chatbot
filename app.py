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
# =========================
# 💬 질의 함수 (RAG 기반)
# =========================
def query(question: str) -> str:
    try:
        # ✅ 질문을 벡터로 임베딩
        q_vec = embed_text(question)
        df = fetch_topk_chunks(q_vec, 5)

        if df.empty:
            return "유사한 청크를 찾지 못했습니다. 질문을 더 구체적으로 입력해 주세요."

        # ✅ 상위 3개의 청크를 모두 컨텍스트로 결합
        top_n = min(3, len(df))
        combined_context = "\n\n---\n\n".join(
            f"[{i+1}] {row['text']}" for i, row in df.head(top_n).iterrows()
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

    except Exception as e:
        return f"⚠️ 오류가 발생했습니다: {e}"

# =========================
# 🎨 UI 구성 (Streamlit)
# =========================
st.markdown("""
<style>
body { background-color: #f3f4f6; font-family: Pretendard, Inter, sans-serif; }

/* 채팅 박스 */
.chat-box {
    background: white; border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    width: 100%; max-width: 480px; margin: auto; padding: 16px;
}

/* 타이틀 박스 - 높이 80% */
.chat-header {
    background-color: #2563eb; color: white;
    padding: 8px 14px;          /* 위아래 padding 줄임 (80%) */
    border-radius: 8px; margin-bottom: 10px;
    text-align: center;
}

/* 말풍선 스타일 */
.user-bubble {
    background-color: #2563eb; color: white;
    border-radius: 20px 20px 0 20px;
    padding: 10px 14px; margin-bottom: 8px;
    margin-left: auto; max-width: 80%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.bot-bubble {
    background-color: #e5e7eb; color: #111827;
    border-radius: 20px 20px 20px 0;
    padding: 10px 14px; margin-bottom: 8px;
    margin-right: auto; max-width: 80%;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

/* 입력창 */
.chat-input {
    border: 2px solid #2563eb; border-radius: 20px;
    padding: 10px 16px; width: 100%;
    font-size: 15px; outline: none;
    box-shadow: 0 2px 6px rgba(37,99,235,0.25);
}

/* 보내기 버튼 중앙 정렬 */
.send-btn {
    display: flex;
    justify-content: center;  /* 가운데 정렬 */
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


with st.container():
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    st.markdown("<div class='chat-header'><h3>보험약관 챗봇</h3><p>NHLife | Made by 태훈,현철</p></div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        bubble_class = "user-bubble" if msg["role"] == "user" else "bot-bubble"
        st.markdown(f"<div class='{bubble_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("", placeholder="상품에 대해 궁금한 점 질문해주세요.", label_visibility="collapsed")
        st.markdown("<div class='send-btn'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("📎 보내기")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# 💬 입력 처리 (Streamlit 이벤트)
# =========================
if submitted and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 답변 생성 전 파란색 박스 표시
    with st.spinner("💙 약관 내용을 검토 중입니다..."):
        st.session_state.messages.append({"role": "assistant", "content": "답변을 생성 중입니다. 잠시만 기다려주세요 💬"})
        st.rerun()

if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["content"].startswith("답변을 생성 중"):
    # 실제 답변 생성 (generate_answer → query 로 교체)
    question = st.session_state.messages[-2]["content"]
    answer = query(question)
    st.session_state.messages[-1] = {"role": "assistant", "content": answer}
    st.rerun()