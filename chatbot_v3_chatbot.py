### 서버용
import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import shutil

# OpenAI API 키 설정
openApiKey = ''

# Streamlit 앱 설정
st.set_page_config(page_title="Strawberry-Seolhyang QA", layout="centered")
st.title("🍓 설향 재배 전문 챗봇 🍓")
st.markdown("설향 재배 관련 질문을 무엇이든 입력해주세요!")

# FAISS 인덱스 업로드 기능 추가
st.sidebar.header("📂 FAISS 인덱스 업로드")
uploaded_faiss = st.sidebar.file_uploader("FAISS 인덱스 파일 (.faiss)", type=["faiss"])
uploaded_pkl = st.sidebar.file_uploader("FAISS 메타데이터 파일 (.pkl)", type=["pkl"])

# 업로드된 파일을 저장할 경로
faiss_dir = "./uploaded_faiss"
faiss_path = os.path.join(faiss_dir, "index.faiss")
pkl_path = os.path.join(faiss_dir, "index.pkl")

# 업로드된 파일 저장 처리
if uploaded_faiss is not None and uploaded_pkl is not None:
    os.makedirs(faiss_dir, exist_ok=True)

    with open(faiss_path, "wb") as f:
        f.write(uploaded_faiss.read())
    with open(pkl_path, "wb") as f:
        f.write(uploaded_pkl.read())

    st.sidebar.success("✅ FAISS 인덱스 & 메타데이터 업로드 완료!")

# FAISS 로드
if os.path.exists(faiss_path) and os.path.exists(pkl_path):
    embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)
    docsearch = FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
    retriever = docsearch.as_retriever(search_kwargs={"k": 1})
    chat_model = ChatOpenAI(openai_api_key=openApiKey, model="gpt-4o-mini")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )
    st.sidebar.success("✅ FAISS 인덱스가 정상적으로 로드되었습니다!")
else:
    st.sidebar.warning("⚠️ FAISS 인덱스(.faiss)와 메타데이터(.pkl)를 모두 업로드해야 합니다.")

# 설향 딸기 재배 전문가에 맞는 프롬프트 템플릿 정의
prompt_template = """
당신은 설향 딸기 재배 전문가입니다.
사용자가 물어보는 질문에 대해 설향 재배와 관련된 전문적인 지식으로 친절하고 자세하게 답변해 주세요.

사용자가 한국어로 질문을 하면, 한국어로만 답변해주세요.

사용자가 일본어로 질문을 하면, 우선 한국어로 답변을 생성한 후 해당 답변을 일본어로 번역하여 일본어 답변을 함께 제공해 주세요.

사용자가 영어로 질문을 하면, 우선 한국어로 답변을 생성한 후 해당 답변을 영어로 번역하여 영어 답변을 함께 제공해 주세요.

답변에 온도 범위를 표시하는 '~'은 '-'로 변경해서 답해주세요.
검색되지 않은 내용에 대해서 임의로 생성해서 답변하지 말아주세요.

한국어 질문 시 절대로 다른 언어와 함께 답변하지 마세요
영어 질문 시 반드시 해당 영어로 답변하세요
일본어 질문 시 반드시 해당 일본어로 답변하세요

질문: {question}

답변:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template,
)
def get_chatbot_response(user_query):
    formatted_prompt = prompt.format(question=user_query)
    response = qa_chain.run(formatted_prompt)
    return response

# 추천 질문 목록 (최초 질문 리스트)
if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = [
        "설향 딸기의 병충해 예방 방법은 무엇인가요?",
        "탄저병 발생 시 사용해야하는 약제를 추천해주세요.",
        "How do you manage fertilizer for Seolhyang strawberries?",
        "ソルヒャンいちごの最適な成長温度は何度ですか？",
        "설향 딸기 재배 시 중요한 요소는 무엇인가요?",
        "설향 딸기의 수확 시기는 언제인가요?",
        "설향 딸기 재배 시 흔한 실수는 무엇인가요?"
    ]

# 세션 상태 초기화 (대화 기록 유지)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pre_filled_query" not in st.session_state:
    st.session_state.pre_filled_query = ""

# 기존 대화 이력 표시
for question, answer in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(f"**사용자:** {question}")
    with st.chat_message("assistant"):
        st.write(f"**설향 전문가:** {answer}")

# 추천 질문을 질문 입력 창 바로 위에 배치 및 가로 2줄 정렬 (여백 추가, 버튼 크기 조절)
st.write("### 추천 질문")
st.markdown("""
<style>
    .stButton>button {
        margin: 8px;
        padding: 12px;
        width: 95%;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

cols = st.columns([2, 2])  # 좌우 여백 확보

for i in range(0, 4, 2):  # 4개씩 보이도록 설정
    with cols[0]:
        if st.button(st.session_state.recommended_questions[i], key=f"rec_{i}", use_container_width=True):
            st.session_state["pre_filled_query"] = st.session_state.recommended_questions[i]
            clicked_question = st.session_state.recommended_questions.pop(i)
            st.session_state.recommended_questions.append(clicked_question)
            st.rerun()
    with cols[1]:
        if st.button(st.session_state.recommended_questions[i + 1], key=f"rec_{i+1}", use_container_width=True):
            st.session_state["pre_filled_query"] = st.session_state.recommended_questions[i + 1]
            clicked_question = st.session_state.recommended_questions.pop(i + 1)
            st.session_state.recommended_questions.append(clicked_question)
            st.rerun()

# 사용자 입력 받기
query = st.text_input("질문을 입력하세요", value=st.session_state.pre_filled_query)

# 질문이 입력되면 챗봇 응답을 생성하고 기록
if query:
    response = get_chatbot_response(query)
    
    # 대화 기록 저장
    st.session_state.chat_history.append((query, response))
    st.session_state["pre_filled_query"] = ""  # 입력 후 초기화
    
    # 입력 즉시 화면 업데이트
    st.rerun()
