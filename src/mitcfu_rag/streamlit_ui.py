import os
import time
import random
import json
import streamlit as st
import requests

# from fakta_chat.config import RAG

# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationChain

current_dir = os.path.dirname(os.path.abspath(__file__))
relative_img_path = os.path.join(current_dir, "faktalink_icon.png")
STREAMING_ENDPOINT = "http://ai-p301:5011"

version = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# @st.cache_resource()
# def load_model():
#     return RAG()

# FAKTARAG = load_model()
roles_to_ignore = ["resetter", "summarizer", "keeper of sources"]

if not st.session_state:
    chat_history = None


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": greeting}]
    global chat_history
    chat_history = None
    # for message in st.session_state.messages:
    #    if not message["role"] in roles_to_ignore:
    #        with st.chat_message(message["role"]):
    #            st.write(message["content"])
    # st.session_state.messages = [{"role": "resetter", "content": None}]
    # for message in st.session_state.messages:
    #     if not message["role"] == "summarizer":
    #         with st.chat_message(message["role"]):
    #             st.write(message["content"])


def summary_references():
    references = []
    global chat_history
    chat_history = st.session_state.messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            references.append(message)
        elif message["role"] == "keeper of sources":
            references.append({"role": "assistant", "content": message["content"]})
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Her findes en opsammling af de brugte kilder fra faktachat. \n \
                                  Svar fra Faktachat kan også indeholde information uden for de givne artikler",
        }
    ] + references
    return chat_history


def chat():
    st.session_state.messages = chat_history


st.sidebar.button("New Chat", on_click=clear_chat_history)
# st.sidebar.button('Chat', on_click=chat) #TODO: does not function with global variable
st.sidebar.button("Opsammling relevante kilder", on_click=summary_references)

st.image(relative_img_path, width=150)

# memoryforchat=ConversationBufferMemory()
# convo=ConversationChain(memory=memoryforchat,llm=chat,verbose=True)

greeting = (
    "Hej 👋 Jeg er MitCFU-RAG og jeg kan hjælpe dig med at finde information om materialer\
              fra MitCFU. Men indtil videre er jeg vist stadig mest en kopi af FaktaChat. \
              \n\nHvad kan jeg hjælpe dig med?"
)


def decode(input):
    try:
        if isinstance(input, str):
            return input
        else:
            return input.decode("utf-8")
    except UnicodeDecodeError as e:
        return input.decode("utf-8", errors="ignore")


def gen_wrapper(stream):
    # for item in stream.iter_content(chunk_size=None, decode_unicode=True):
    #     for i in item:
    #         yield i

    for item in stream.iter_content(chunk_size=None, decode_unicode=True):
        decoded_item = decode(item)
        obj = json.loads(decoded_item.replace("data:", ""))
        if not obj.get("token", {}).get("text", {}) == "</s>":
            yield obj.get("token", {}).get("text", {})


# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "greeting", "content": greeting}]

# Create chat history
# if "chat_history" not in st.session_state:
#    st.session_state.chat_history=[]
# else:
#    for message in st.session_state.chat_history:
#        memoryforchat.save_context({"input":message["human"]}, {"outputs":message["AI"]})

# with st.chat_message("assistant"):
# st.write_stream(gen_wrapper([' ' + x for x in greeting.split(' ')]))
# st.write_stream((str(' ') + x for x in greeting.split(' ')))

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] not in roles_to_ignore:
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
            "Lad mig finde relevant information på faktalink ...",
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
            response = st.write_stream(gen_wrapper(response_stream))
            st.session_state.messages.append({"role": "assistant", "content": response})
        if references:
            ref_text = ""
            st.write("Kilder:\n")
            stream_text = "Kilder:\n"
            ref_text += "Kilder:\n"
            seen_references = set()
            for ref in references:
                if not ref.article_link in seen_references:
                    st.write_stream(gen_wrapper(ref.article_link.split("\n")))
                    stream_text += "\n\n" + ref.article_link
                    ref_text += (
                        "\n\n"
                        + ref.article_link
                        + ' - Artikel: "'
                        + ref.article_headline
                        + '"\n\n'
                    )
                seen_references.add(ref.article_link)
            st.session_state.messages[-1]["content"] = (
                st.session_state.messages[-1]["content"] + stream_text
            )
            # st.session_state.messages.append({"role": "keeper of sources", "content": ref_text})

# Streamed response emulator
# def response_generator(i: int):
#     response_str = FAKTARAG.get_response(st.session_state.messages)
#     for word in response_str.split(" "):
#         if "..." in word:
#             print(word)
#             yield word + " "
#             time.sleep(0.6)
#         else:
#             yield word + " "
#             time.sleep(0.05)


# Uncomment this to enable summarizer
# if i>0:
# generate summary and append to chat history
#    summary = FAKTARAG.get_summary(st.session_state.messages)
#    st.session_state.messages.append({"role": "summarizer", "content": summary})
