import streamlit as st
import os
from chatphd import (
    get_all_documents,
    get_message_stream
)

def init_session_state():
    if 'is_init' not in st.session_state or not st.session_state.is_init:
        st.session_state.selected_document = None
        st.session_state.messages = []
        st.session_state.is_init = True
        st.session_state.chat_started = False

def start_chat():
    # Just set the initial state without displaying anything
    st.session_state.chat_started = True
    initial_message = "What is this paper about? Give me a brief overview."
    st.session_state.messages = [
        {"role": "user", "content": initial_message},
        {"role": "assistant", "content": None}  # Placeholder for the response
    ]

def init_ui():
    # Set page title and favicon
    st.set_page_config(
        page_title="chatphd",
        page_icon="📚"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            background: linear-gradient(to right, #4776E6, #8E54E9);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(to right, #8E54E9, #4776E6);
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .title {
            background: linear-gradient(to right, #4776E6, #8E54E9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
        }
        </style>
        """, unsafe_allow_html=True)

    # Use custom styled title
    st.markdown("<h1 class='title'>Chat with a paper</h1>", unsafe_allow_html=True)

    init_session_state() 

    st.selectbox(
        "Select a paper",
        get_all_documents(),
        index=None, 
        placeholder="No paper selected",
        key="selected_document",
        on_change=on_doc_change, 
        format_func=lambda doc: doc.full_name
    )

    # Show the big button when document is selected but chat hasn't started
    if st.session_state.selected_document is not None and not st.session_state.chat_started:
        st.button("Start chatting 🚀", on_click=start_chat, use_container_width=True)
    elif st.session_state.selected_document is None:
        st.info("🔍 Please select a paper to start chatting.")
    
    # Only show chat interface if chat has been started
    if st.session_state.chat_started:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant" and message["content"] is None:
                        # This is our initial assistant message that needs to be streamed
                        stream = get_message_stream(
                            [{"role": "user", "content": st.session_state.messages[0]["content"]}],
                            st.session_state.selected_document
                        )
                        response = st.write_stream((chunk.delta.text for chunk in stream if chunk.type == "content_block_delta"))
                        st.session_state.messages[i]["content"] = response
                    else:
                        st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the paper."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]
                
                stream = get_message_stream(messages, st.session_state.selected_document)
                response = st.write_stream((chunk.delta.text for chunk in stream if chunk.type == "content_block_delta"))
            st.session_state.messages.append({"role": "assistant", "content": response})

def on_doc_change(): 
    st.session_state.messages = []
    st.session_state.chat_started = False
    print(st.session_state.selected_document)
