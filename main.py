from utils import get_vector_store, getLLM, getPromptTemplate, printResponse
from langchain.chains import RetrievalQA

pc_index_name = "icyco-ai-assistant"             # Define the vector store index
pdf_files = ["about.pdf", "products.pdf"]        # Define pre PDF files

vector_store = get_vector_store(pc_index_name, pdf_files)     # Initialize vector store
llm = getLLM()
prompt_template = getPromptTemplate()

print("\nInitializing QA RAG system...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template},
    verbose=True,
)

print("Yes, The RAG system is successfully initialized.")
query = input("\nAsk your question: ")
response = qa_chain.invoke({"query": query})  # Process the query
print("processing...")

printResponse(response)