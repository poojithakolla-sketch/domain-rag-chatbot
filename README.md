# Domain-Specific RAG Chatbot for PDF Question Answering

## 📌 Project Overview

The Domain-Specific RAG Chatbot is a document question-answering application that allows users to upload PDF documents and ask questions based on their content.

The system retrieves relevant passages from the uploaded documents and uses a Large Language Model to generate grounded answers.

It also displays the source document and page number for the retrieved information.

---

## 🎯 Objective

The objective of this project is to build a reliable question-answering system over uploaded PDF documents.

The chatbot should:

- Extract text from PDF documents
- Split documents into smaller chunks
- Generate embeddings for the chunks
- Store embeddings using FAISS
- Retrieve relevant document passages
- Generate answers using the retrieved context
- Display source document and page number
- Avoid inventing information that is not available in the documents

---

## 🔄 System Workflow

```text
Upload PDF
     ↓
Extract Text
     ↓
Split into Chunks
     ↓
Generate Embeddings
     ↓
Store in FAISS
     ↓
User Question
     ↓
Retrieve Relevant Chunks
     ↓
Groq LLM
     ↓
Generate Grounded Answer
     ↓
Display Answer + Source/Page