import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_pipeline import RAGPipeline
from prompt import create_prompt


# Load environment variables
load_dotenv()


# Page configuration
st.set_page_config(
    page_title="Domain-Specific RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# Title
st.title("🤖 Domain-Specific RAG Chatbot")

st.write(
    "Upload PDF documents and ask questions "
    "based on their content."
)


# Initialize RAG pipeline
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()


# Processing status
if "processed" not in st.session_state:
    st.session_state.processed = False


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Get Groq API key
api_key = os.getenv("GROQ_API_KEY")


# Create Groq client
if api_key:
    client = Groq(api_key=api_key)
else:
    client = None


# Sidebar
with st.sidebar:

    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True
    )

    if process_button:

        if not uploaded_files:

            st.warning(
                "Please upload at least one PDF."
            )

        else:

            with st.spinner(
                "Processing documents..."
            ):

                chunk_count = (
                    st.session_state.rag
                    .process_documents(uploaded_files)
                )

            st.session_state.processed = True

            st.success(
                f"Successfully processed {chunk_count} chunks."
            )

    st.divider()

    # Show uploaded files
    if uploaded_files:

        st.subheader("Uploaded Files")

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

    st.divider()

    # Clear chat
    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            st.markdown(
                "**📚 Sources:**"
            )

            for source in message["sources"]:

                st.write(
                    f"📄 {source['source']} "
                    f"— Page {source['page']}"
                )


# Chat input
question = st.chat_input(
    "Ask a question about your PDF..."
)


if question:

    # Check whether documents were processed
    if not st.session_state.processed:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()


    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # Display user message
    with st.chat_message("user"):

        st.markdown(question)


    # Retrieve relevant chunks
    with st.spinner(
        "Searching your documents..."
    ):

        results = (
            st.session_state.rag
            .retrieve(
                question,
                top_k=5
            )
        )


    # No relevant results
    if not results:

        answer = (
            "I could not find this information "
            "in the uploaded documents."
        )

        sources = []


    else:

        # Build context
        context_parts = []

        for result in results:

            context_parts.append(
                f"""
Source: {result['source']}
Page: {result['page']}

{result['text']}
"""
            )

        context = "\n\n".join(
            context_parts
        )


        # Create RAG prompt
        final_prompt = create_prompt(
            context,
            question
        )


        # Check Groq API key
        if client is None:

            answer = (
                "Groq API key is missing. "
                "Please add GROQ_API_KEY "
                "to the .env file."
            )

        else:

            try:

                response = client.chat.completions.create(

                    model="openai/gpt-oss-120b",

                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a document "
                                "question-answering assistant. "
                                "Answer ONLY using the "
                                "provided context. "
                                "Do not use outside knowledge. "
                                "Do not invent information."
                            )
                        },
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],

                    temperature=0,

                    max_tokens=1024
                )


                answer = (
                    response
                    .choices[0]
                    .message
                    .content
                )


            except Exception as e:

                answer = (
                    "An error occurred while "
                    "generating the answer."
                )

                st.error(
                    str(e)
                )


        sources = results


    # Display assistant answer
    with st.chat_message(
        "assistant"
    ):

        st.markdown(answer)


        # Display sources
        if sources:

            st.markdown(
                "**📚 Sources:**"
            )


            displayed_sources = set()


            for source in sources:

                source_key = (
                    source["source"],
                    source["page"]
                )


                if source_key not in displayed_sources:

                    st.write(
                        f"📄 {source['source']} "
                        f"— Page {source['page']}"
                    )

                    displayed_sources.add(
                        source_key
                    )


    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )