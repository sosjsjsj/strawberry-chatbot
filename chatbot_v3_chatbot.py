import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Streamlit 앱 설정
st.set_page_config(page_title="Strawberry-Seolhyang QA", layout="centered")

# OpenAI API 키 설정
openApiKey = 'personal_api_key'

# FAISS 인덱스 로드
embeddings = OpenAIEmbeddings(openai_api_key=openApiKey)
docsearch = FAISS.load_local("./sh_txt_vec", embeddings, allow_dangerous_deserialization=True)

# OpenAI 모델 초기화
chat_model = ChatOpenAI(openai_api_key=openApiKey, model="gpt-4o-mini")

# 메모리 설정
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# RetrievalQA 체인 설정
qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
    memory=memory
)

# ✅ 프롬프트 템플릿
prompt_template = """
당신은 설향 딸기 재배 전문가입니다.
사용자가 물어보는 질문에 대해 설향 재배와 관련된 전문적인 지식으로 친절하고 자세하게 답변해 주세요.

사용자가 한국어로 질문을 하면, 한국어로만 답변해주세요.
사용자가 일본어로 질문을 하면, 우선 한국어로 답변을 생성한 후 해당 답변을 일본어로 번역하여 일본어 답변을 함께 제공해 주세요.
사용자가 영어로 질문을 하면, 우선 한국어로 답변을 생성한 후 해당 답변을 영어로 번역하여 영어 답변을 함께 제공해 주세요.

답변에 온도 범위를 표시하는 '~'은 '-'로 반드시 변경해서 답해주세요.
검색되지 않은 내용에 대해서 임의로 생성해서 답변하지 말아주세요.

한국어 질문 시 절대로 다른 언어와 함께 답변하지 마세요
영어 질문 시 반드시 영어로 답변하세요
일본어 질문 시 반드시 일본어로 답변하세요
답변에 '~'이 포함된다면 '-'로 반드시 변경해서 답해주세요.

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

st.markdown("""
    <style>
    .title {
        font-size: 30px !important; /* 제목 크기 조절 (기본 32px → 28px) */
        font-weight: bold;
        text-align: left;
    }
    .recommend-title {
        font-size: 28px; /* 추천 질문 크기 줄이기 */
        font-weight: bold;
        text-align: left;
        margin-bottom: 20px;
    }
    .message-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin: 10px 0;
    }
    .message-label {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 3px;
        padding: 5px 10px;
        border-radius: 8px;
        display: inline-block;
    }
    .user-label {
        color: black;
    }
    .assistant-label {
        color: black;
    }
    .user-message {
        background-color: #FFD1D1; /* 연한 빨강 */
        padding: 12px;
        border-radius: 12px;
        max-width: 75%;
        display: inline-block;
    }
    .assistant-message {
        background-color: #D0E3FF; /* 연한 파랑 */
        padding: 12px;
        border-radius: 12px;
        max-width: 75%;
        display: inline-block;
    }
    .icon {
        font-size: 16px;
        margin-right: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# ✅ Streamlit UI
st.markdown('<p class="title">🍓 말하는 딸기 🍓</p>', unsafe_allow_html=True)
st.markdown("설향 재배 관련 질문을 무엇이든 입력해주세요!")
st.write("")  # 한 줄 여백 추가
st.write("")  # 두 줄 여백 추가

# 추천 질문 목록
if "recommended_questions" not in st.session_state:
    st.session_state.recommended_questions = [
        "설향의 특징을 모두 알려주세요.",
        "설향의 새잎이 황록색으로 변했어요.",
        "설향 재배 시 탄저병이 발생하면 사용해야하는 약제를 추천해주세요.",
        "How do you manage fertilizer for Seolhyang strawberries?",
        "ソルヒャンいちごの最適な成長温度は何度ですか？",
        "설향 정식 시 참고해야할 점을 알려주세요.",
        "설향의 과번무 시 생기는 문제점을 알려주세요.",
        "설향 재배 시 탄산가스 관리를 어떻게 해야하나요?",
        "설향 재배 시 중요한 요소는 무엇인가요?",
        "1월 설향 재배 시 잿빛곰팡이병 방제를 위해 어떤 농작업이 필요한가요?",
        "10월 설향 재배 시 주의점을 알려주세요",
        "설향의 수확 시기는 언제인가요?",
        "설향 재배 시 흔한 실수는 무엇인가요?"
    ]

# 세션 상태 초기화 (대화 기록 유지)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pre_filled_query" not in st.session_state:
    st.session_state.pre_filled_query = ""

# ✅ 기존 대화 이력 표시 (사용자는 오른쪽, 챗봇은 왼쪽)
for question, answer in st.session_state.chat_history:
    st.markdown(f"""
    <div class="message-container user-container">
        <div class="message-label"><b>😎사용자😎</b></div>
        <div class="user-message">{question}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="message-container assistant-container">
        <div class="message-label"><b>🍓설향 전문가🤖</b></div>
        <div class="assistant-message">{answer}</div>
    </div>
    """, unsafe_allow_html=True)



# 추천 질문을 질문 입력 창 위에 배치
st.markdown('<p class="recommend-title">📌 추천 질문 (클릭하세요!)</p>', unsafe_allow_html=True)  
cols = st.columns([2, 2])

for i in range(0, 4, 2):
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

st.write("")  # 한 줄 여백 추가

# 사용자 입력 받기
query = st.text_input("질문을 입력하세요", value=st.session_state.pre_filled_query)

# 질문이 입력되면 챗봇 응답을 생성하고 기록
if query:
    response = get_chatbot_response(query)
    
    # 대화 기록 저장
    st.session_state.chat_history.append((query, response))
    st.session_state["pre_filled_query"] = ""  # 입력 후 초기화

    # 사용자 메시지 (오른쪽)
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message"><span class="icon">😃</span>{query}</div>', unsafe_allow_html=True)

    # 챗봇 메시지 (왼쪽 - 파란색)
    with st.chat_message("assistant"):
        st.markdown(f'<div class="assistant-message"><span class="icon">🏡</span>{response}</div>', unsafe_allow_html=True)

    # 입력 즉시 화면 업데이트
    st.rerun()
