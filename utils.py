import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
import time  
import os  
from langchain_google_genai import GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec  
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
'''
This module contains utility functions for text processing.
'''
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def getTextSplitter():
    # Create an instance of RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return textsplitter

def create_chunks(file_path):
    loader = PyPDFLoader(f"Documents/{file_path}")
    documents = loader.load()
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    chunked_docs = getTextSplitter().split_documents(documents)
    return chunked_docs

def printResponse(response):
    print("\n=== Answer ===")
    print(response["result"])

    print("=== Source Documents ===")
    for doc in response["source_documents"]:
        print(f"\n{doc.metadata['title']}, Page no: {doc.metadata['page_label']}:")
        print(f"Document content: {doc.page_content}...")

def getEmbeddingModel():
    print("Initializing embedding model...")
    
    # Initialize the embedding model (you can replace FastEmbedEmbeddings with your chosen model)
    # This model will be used to convert text into vector embeddings (numerical representations)
    embedding_model = FastEmbedEmbeddings()
    
    # Return the initialized embedding model to be used later in the code
    return embedding_model

def add_documents_to_vector_store(vector_store, pdfs):
    # Initialize an empty list to hold text chunks from the PDFs
    chunks = []
    
    # Loop through each PDF in the provided list of PDFs
    for pdf in pdfs:
        # Create chunks from the PDF (create_chunks function implemented in step 2)
        # Chunks are small pieces of the document that can be embedded for efficient search
        chunks.extend(create_chunks(pdf))
    
    # Add the extracted text chunks to the vector store
    # The vector store will automatically create embeddings and index the chunks for fast similarity search
    vector_store.add_documents(chunks)

def create_vector_store(pc, pc_index_name, pdf_files, embedding_model):
    pc.create_index(
        name=pc_index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(pc_index_name).status['ready']:  # Wait for the index to be ready
        time.sleep(1)
    print(f"Successfully created vector store index '{pc_index_name}'. Adding pdf files to new index...")
    index = pc.Index(pc_index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
    add_documents_to_vector_store(vector_store, pdf_files)
    return vector_store

def get_vector_store(pc_index_name, pdf_files):
    print("Initializing pinecone vector store...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding_model = getEmbeddingModel()
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if pc_index_name not in existing_indexes:
        print(f"Vector store index '{pc_index_name}' not found. Create new...")
        create_vector_store(pc, pc_index_name, pdf_files, embedding_model)
    index = pc.Index(pc_index_name)
    return PineconeVectorStore(index=index, embedding=embedding_model)

def getLLM():
    print("Initializing LLM...")
    # Initialize the LLM (Google Generative AI)
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
    return llm

def getPromptTemplate():
    print("Initializing prompt template...")
    prompt_template = PromptTemplate(
        template="""
            You are a helpful, friendly, and engaging AI assistant for **Icyco**, an ice cream shop. 
            Your job is to answer user questions related to Icyco’s products, services, events, or company info in a way that is both informative and delightful.

            Use the following retrieved documents to provide an accurate and engaging response.

            - If the user’s question is **not related to Icyco**, politely respond with:  
              *"I’m here to help with questions about Icyco only — let me know if there’s something specific you’re curious about!"*

            - If the user’s question **is related to Icyco** but you **cannot find the answer** in the provided documents, say:  
              *"That’s a great question about Icyco, but I couldn’t find the answer in the info I have. You might want to reach out to Icyco directly for the most up-to-date details!"*

            Your tone should be:
            - Friendly and conversational 🧁  
            - Engaging and easy to understand  
            - Accurate — don’t make up any information

            If it fits naturally, feel free to show enthusiasm, add a touch of personality, or relate to the excitement around ice cream!

            ---  
            Context:  
            {context}

            user: {question}
            Assistant:
        """,
        input_variables=["context", "question"],
    )
    return prompt_template