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


def on_doc_change(): 
    st.session_state.messages = []
    print(st.session_state.selected_document)
    
def init_ui():
    st.title("Chat with a paper")

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

    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
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
