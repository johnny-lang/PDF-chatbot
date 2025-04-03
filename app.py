from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader    
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
import gradio as gr
import os

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Tải model GGUF 
def download_gguf_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("./models", exist_ok=True)
        print("Downloading GGUF model...")
        import requests
        url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded successfully!")

# Load LLM từ file GGUF đã cache
def get_llm():
    download_gguf_model()  # Đảm bảo model đã được tải
    
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_ctx=2048,       
        n_threads=4,      
        max_tokens=256,    
        temperature=0.7,
    )
    return llm

# Load Embedding model với cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        cache_folder="./embedding_models" 
    )

# Phần xử lý tài liệu
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def text_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def vector_db(texts):
    embeddings = get_embeddings()
    return Chroma.from_documents(texts, embeddings)

def get_retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    db = vector_db(chunks)
    return db.as_retriever(search_kwargs={"k": 3})

# QA Chain
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

# Gradio Interface 
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF", type="filepath"), 
        gr.Textbox(label="Ask a question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Application (GGUF + Cache)",
    description="Chat with your PDF using locally cached Mistral-7B GGUF model",
)

if __name__ == "__main__":
    download_gguf_model()
rag_application.launch(server_name="127.0.0.1", server_port=7860, share=True)