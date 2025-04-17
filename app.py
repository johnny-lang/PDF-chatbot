import os
import dotenv
import warnings
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings

import gradio as gr
from openai import OpenAI

warnings.filterwarnings("ignore")
dotenv.load_dotenv()

# Khởi tạo client OpenAI kiểu tương thích Cohere API
client = OpenAI(
    api_key=os.getenv("COHERE_API_KEY"),
    base_url="https://api.cohere.ai/compatibility/v1"
)

# ---- Embedding Class cho Cohere API ----
class CohereEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = client.embeddings.create(
            input=texts,
            model="embed-v4.0",
            encoding_format="float"
        )
        return [d.embedding for d in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ---- Fireworks LLM ----
def get_llm():
    return ChatOpenAI(
        openai_api_base="https://api.fireworks.ai/inference/v1",
        openai_api_key=os.getenv("FIREWORKS_API_KEY"),
        model_name="accounts/fireworks/models/llama-v3p1-405b-instruct",
        temperature=0.1
    )

# ---- Load PDF ----
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# ---- Text splitting ----
def text_splitter(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)

# ---- Vector DB ----
def vector_db(texts):
    embeddings = CohereEmbedding()
    return Chroma.from_documents(texts, embeddings)

# ---- Create retriever ----
def get_retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    db = vector_db(chunks)
    return db.as_retriever(search_kwargs={"k": 3})

# ---- Retrieval QA ----
def retriever_qa(file, query):
    llm = get_llm()
    retriever = get_retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
    )
    response = qa({"query": query})
    return response['result']

# ---- Gradio UI ----
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"),
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Application (Fireworks + Cohere)",
    description="Chat với PDF bằng Fireworks LLM và Cohere Embedding API",
)

if __name__ == "__main__":
    rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)
