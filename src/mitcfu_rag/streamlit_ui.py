import os
import time
import random
import json
import streamlit as st
import requests

from mitcfu_rag.tools.llm_formatting import gen_wrapper, GEMMA_3_12B, MIXTRAL_8X7B

# from fakta_chat.config import RAG

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_img_path = os.path.join(current_dir, "faktalink_icon.png")
STREAMING_ENDPOINT = "http://ai-p301:5011"

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

st.image(relative_img_path, width=150)


greeting = (
    "Hej 👋 Jeg er MitCFU-RAG og jeg kan hjælpe dig med at finde information om materialer\
              fra MitCFU. Men indtil videre er jeg vist stadig mest en kopi af FaktaChat. \
              \n\nHvad kan jeg hjælpe dig med?"
)

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
            references = []
            response_stream = requests.post(
                STREAMING_ENDPOINT,
                json={"messages": st.session_state.messages},
                stream=True,
            )
            response = st.write_stream(gen_wrapper(response_stream, model_name=GEMMA_3_12B))
            st.session_state.messages.append({"role": "assistant", "content": response})
