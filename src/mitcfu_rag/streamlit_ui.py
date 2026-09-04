import os
import random
import streamlit as st
from openai import OpenAI

from mitcfu_rag.config import DEFAULT_MODEL

STREAMING_ENDPOINT = os.environ.get("MITCFU_UI_STREAM_URL", "http://localhost:5000/v1/chat/completions")
client = OpenAI(base_url=STREAMING_ENDPOINT, api_key="unused")
version = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if not st.session_state:
    chat_history = None


def clear_chat_history():
    st.session_state.messages = []
    global chat_history
    chat_history = None


def chat():
    st.session_state.messages = chat_history


st.sidebar.button("New Chat", on_click=clear_chat_history)


greeting = "Hej 👋 Jeg er MitCFU-RAG og jeg kan hjælpe dig med at finde information om materialer\
              fra MitCFU. Men indtil videre er jeg vist stadig mest en kopi af FaktaChat. \
              \n\nHvad kan jeg hjælpe dig med?"

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "greeting", "content": greeting}]

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(message["role"], avatar="🤓"):
            st.write(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# React to user input
if prompt := st.chat_input("Indsæt dit spørgmål her ..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar="🤓"):
        fillers = [
            "Lad mig finde relevant information i mitcfu ...",
            "Lad mig se...",
            "Et øjeblik...",
            "Vent lige...",
            "Hmm, lad mig finde noget...",
        ]
        with st.spinner(random.choice(fillers)):
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=st.session_state.messages,
                stream=True,
            )
            response = st.write_stream(c.choices[0].delta.content for c in stream if c.choices[0].delta.content)
            st.session_state.messages.append({"role": "assistant", "content": response})
