import streamlit as st
from openai import OpenAI
#import certifi
# import os

st.set_page_config(page_title="Streamlit Chat", page_icon="💬")
st.title("Chatbot")

if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False
if "user_message_count" not in st.session_state:
    st.session_state.user_message_count = 0
if "feedback_shown" not in st.session_state:
    st.session_state.feedback_shown = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_complete" not in st.session_state:
    st.session_state.chat_complete = False
if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True

def complete_chat():
    st.session_state.chat_complete = True

if not st.session_state.setup_complete:
    st.subheader('Personal information', divider='blue')

    st.text_input("Name", key="name", max_chars=40, placeholder="Enter your name:")
    st.text_area("Experience", key="experiences", max_chars=200, placeholder="Describe your experience:")
    st.text_area("Skills", key="skills", max_chars=200, placeholder="List your skills")

    st.subheader('Company and Position', divider='blue')

    if "level" not in st.session_state:
        st.session_state["level"] = "Junior"
    if "position" not in st.session_state:
        st.session_state["position"] = "Data Scientist"
    if "company" not in st.session_state:
        st.session_state["company"] = "Amazon"

    col1, col2 = st.columns(2)
    with col1:
        st.radio("Choose level:", options=["Junior", "Mid-level", "Senior"], key="level")

    with col2:
        st.selectbox("Choose a position:", ("Data Scientist", "Data Engineer", "ML Engineer", "BI Analyst", "Financial Analyst"), key='position')

    st.selectbox("Choose a Company:", ("Amazon", "Meta", "SpaceX", "Nestle", "LinkedIn", "Spotify"), key='company')

    if st.button("Start Interview", on_click=complete_setup):
        st.write("Setup complete. Starting interview...")

if st.session_state.setup_complete and not st.session_state.feedback_shown and not st.session_state.chat_complete:

    # Fix SSL cert issue
    # os.environ["SSL_CERT_FILE"] = certifi.where()

    # Initialize client
    client = st.session_state.client

    # Session state defaults
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-5-nano"

    if not st.session_state.messages:
        st.session_state.messages = [
            {
                "role": "system",
                "content": f"You are a kind HR executive named David that interviews an interviewee called {st.session_state['name']} with experience {st.session_state['experiences']} and skills {st.session_state['skills']}. You should interview them for the {st.session_state['level']} {st.session_state['position']} at the company {st.session_state['company']}. The interviewee will be limited to only 4 responses after their greeting. Ask 3 questions and one clarifying question to assess their readiness for the position. Ask the questions one at a time to allow the interview to flow naturally and be personable. Make sure that the questions are HR oriented, and only ask one question per question."
            },
            {
            "role": "assistant",
            "content": f"Welcome! It's great to meet you, {st.session_state['name']}. Let's start with a quick introduction—can you tell me about yourself?"
        }
        ]

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if st.session_state.user_message_count < 4:
        # Chat input
        if prompt := st.chat_input("Your answer:", max_chars=1000):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = client.responses.create(
                    model=st.session_state["openai_model"],
                    input=st.session_state.messages
                )
                assistant_message = response.output_text
                st.markdown(assistant_message)
                st.session_state.messages.append({"role": "assistant", "content": assistant_message})
            st.session_state.user_message_count += 1

    else:
        complete_chat()

if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button("Get Feedback"):
        show_feedback()
        st.write("Fetching feedback...")
    
if st.session_state.feedback_shown:
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    client = st.session_state.client
    feedback_completion = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "system",
                "content": f"""You are an interview performance evaluator. Look over the transcript for the interviewee and provide feedback on their performance. Give a score from 1 to 10.
                Follow this format:
                Overall Score: //Your score
                Feedback: //Here you put your feedback
                Give only the feedback, do not ask any additional questions."""
            },
            {"role": "user", "content": f"This is the interview you need to evaluate. Keep in mind that you are only a tool and shouldn't engage in conversation. {conversation_history}"}
        ]
    )

    st.markdown(feedback_completion.output_text)

    if st.button("Restart Interview", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
