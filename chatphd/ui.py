import streamlit as st
import os
from chatphd import (
    client, 
    get_document_content, 
    set_document, 
    get_document_full_name, 
    get_document_short_name
)

def init_ui():
    st.title("Chat with a document")

    # Add document selector
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = get_document_full_name("fcn")

    def on_doc_change():
        st.session_state.pop("messages", None)  # Reset chat
        short_name = get_document_short_name(st.session_state.selected_document)
        set_document(short_name)

    selected_doc = st.selectbox(
        "Select document",
        [get_document_full_name(doc) for doc in ["fcn", "lns"]],
        key="selected_document",
        on_change=on_doc_change
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
            
            stream = client.messages.create(
                model=os.getenv("MODEL_NAME", "claude-3-5-haiku-20241022"),
                system=get_document_content(),
                messages=messages,
                max_tokens=200,
                stream=True
            )
            
            response = st.write_stream((chunk.delta.text for chunk in stream if chunk.type == "content_block_delta"))
        st.session_state.messages.append({"role": "assistant", "content": response})
