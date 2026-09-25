from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from document_loader import extract_text_from_pdf
from vector_store import VectorStore


class RAGPipeline:

    def __init__(self):

        self.vector_store = VectorStore()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )

    def process_documents(self, uploaded_files):

        all_chunks = []

        for uploaded_file in uploaded_files:

            pages = extract_text_from_pdf(
                uploaded_file
            )

            for page in pages:

                chunks = self.splitter.split_text(
                    page["text"]
                )

                for chunk in chunks:

                    all_chunks.append({
                        "text": chunk,
                        "source": page["source"],
                        "page": page["page"]
                    })

        if all_chunks:

            self.vector_store.create_store(
                all_chunks
            )

        return len(all_chunks)

    def retrieve(self, question, top_k=5):

        return self.vector_store.search(
            question,
            top_k
        )