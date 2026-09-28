import os
import random
import requests
import streamlit as st
from openai import OpenAI

STREAMING_ENDPOINT = os.environ.get("MITCFU_UI_STREAM_URL", "http://localhost:5000/v1/chat/completions")
client = OpenAI(base_url=STREAMING_ENDPOINT, api_key="unused")
version = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _service_base_url() -> str:
    """STREAMING_ENDPOINT points at the chat/completions path; strip that
    suffix to reach the service's other routes, e.g. GET /v1/models."""
    return STREAMING_ENDPOINT.removesuffix("/v1/chat/completions").removesuffix("/chat/completions")


@st.cache_data(ttl=60)
def fetch_available_models() -> list[str]:
    """Model ids this deployment can actually route to (GET /v1/models),
    so the picker below never offers a model the service will 400 on."""
    try:
        response = requests.get(f"{_service_base_url()}/v1/models", timeout=5)
        response.raise_for_status()
        return [entry["id"] for entry in response.json().get("data", [])]
    except requests.RequestException as e:
        st.sidebar.warning(f"Kunne ikke hente modelliste fra servicen: {e}")
        return []


if not st.session_state:
    chat_history = None


def clear_chat_history():
    st.session_state.messages = []
    global chat_history
    chat_history = None


def chat():
    st.session_state.messages = chat_history


st.sidebar.button("New Chat", on_click=clear_chat_history)

available_models = fetch_available_models()
selected_model = st.sidebar.selectbox("Model", available_models) if available_models else None


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
            create_kwargs = {
                "messages": st.session_state.messages,
                "stream": True,
            }
            # Omit `model` entirely when the picker has nothing to offer (the
            # /v1/models fetch failed) so the service falls back to its own
            # configured default instead of us guessing a model id.
            if selected_model:
                create_kwargs["model"] = selected_model
            stream = client.chat.completions.create(**create_kwargs)
            response = st.write_stream(c.choices[0].delta.content for c in stream if c.choices[0].delta.content)
            st.session_state.messages.append({"role": "assistant", "content": response})
