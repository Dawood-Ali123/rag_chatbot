import streamlit as st
from backened import chat_bot,reterive_all_threads,submit_async_task,_run_async
from langchain_core.messages import HumanMessage,AIMessageChunk
import uuid
import queue
import os
import tempfile
from rag  import add_pdf
def generate_thread_id():
    return str(uuid.uuid4())
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(
            thread_id
        )
async def load_conversations(thread_id):
    state = await chat_bot.aget_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get(
        "messages",
        []
    )
async def load_title(thread_id):
    state = await chat_bot.aget_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    return state.values.get("title","New Chat")
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = (_run_async(reterive_all_threads()))
add_thread(st.session_state["thread_id"])
#side bar
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.title("Documents")
uploaded_file=st.sidebar.file_uploader(
    "Upload PDF",
    type=['pdf']
)
if uploaded_file:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )as temp_file:
        temp_file.write(
            uploaded_file.getbuffer()
        )
        temp_path=temp_file.name
    try:
        chunks=add_pdf(
            temp_path,
            uploaded_file.name
        )
        st.sidebar.success(f"{uploaded_file.name} uploaded")
        st.sidebar.info(f"{chunks} chunks added")
    finally:
        os.remove(temp_path)
st.sidebar.title("My Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    title = _run_async(load_title(thread_id))
    if st.sidebar.button(title,key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = _run_async(
            load_conversations(thread_id)
        )
        temp_messages = []
        for message in messages:
            if isinstance(message,HumanMessage):
                role = "user"
            else:
                role = "assistant"
            # Ignore tool messages
            if role == "assistant":
                if not hasattr(message,"content"):
                    continue
            temp_messages.append(
                {
                    "role": role,
                    "content": message.content
                }
            )
        st.session_state["message_history"] = temp_messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


CONFIG = {

    "configurable": {
        "thread_id": st.session_state[
            "thread_id"
        ]
    },

    "metadata": {
        "thread_id": st.session_state[
            "thread_id"
        ]
    },

    "run_name": "chat_turn"
}

# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Type here ---"
)


if user_input:

    # -----------------------------------------------------
    # SAVE USER MESSAGE IN UI HISTORY
    # -----------------------------------------------------

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "user",
            "content": user_input
        }
    )


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_input
        )


    # -----------------------------------------------------
    # ASSISTANT
    # -----------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        def ai_only_stream():
            event_queue = queue.Queue()
            async def run_stream():
                try:
                    async for (message_chunk,metadata) in chat_bot.astream(
                        {
                            "messages": [
                                HumanMessage(
                                    content=user_input
                                )
                            ]
                        },
                        config=CONFIG,
                        stream_mode="messages"):
                        event_queue.put(
                            (
                                "message",message_chunk,metadata
                            )
                        )
                except Exception as e:
                    event_queue.put("error",e)
                finally:

                    event_queue.put(None)
                        
            submit_async_task(run_stream())

            while True:
                event = event_queue.get()
                if event is None:
                    continue
                if event[0] == "done":
                    break

                if event[0] == "error":
                    raise event[1]
                if event[0]=="message":
                    message_chunk = event[1]
                    metadata = event[2]
                if (
                    isinstance(
                        message_chunk,AIMessageChunk
                    )
                    and
                    metadata.get("langgraph_node") == "chat_node"):

                    content = (
                        message_chunk.content
                    )


                    if isinstance(content,str):
                        yield content
                    elif isinstance(
                        content,list
                    ):
                        for item in content:
                            if isinstance(
                                item,dict
                            ):
                                text = item.get(
                                    "text"
                                )
                                if text:
                                    yield text

        ai_message = st.write_stream(
            ai_only_stream()
        )

    st.session_state[
        "message_history"
    ].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )