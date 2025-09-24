# 2. Implement the RAG pipeline
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- 1. Load documents from the 'docs' directory using DirectoryLoader and TextLoader ---
docs_path = "docs/"
# Use DirectoryLoader to load all text files in the specified path
loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()

# --- 2. Split documents into chunks ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(docs)
print(f"Loaded {len(docs)} text files and split into {len(chunked_docs)} chunks.")

# --- 3. Create embeddings and vector store ---
# Use a local, open-source embedding model from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunked_docs, embeddings)
print("Vector store created successfully.")

# --- 4. Set up the local Llama 3.1 model via Ollama ---
# Ensure the Ollama server is running and the llama3.1 model is pulled
llm = Ollama(model="llama3.1")

# --- 5. Define the prompt template ---
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert fraud analyst for a bank. Your task is to analyze banking events
    and determine if they are suspicious based on the provided context.
    
    Answer the user's question based on the following context:
    {context}
    
    Question: {input}
    """
)

# --- 6. Create the RAG chain ---
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

# --- 7. Define functions for different banking fraud types ---
def analyze_card_transaction(transaction_details):
    """Analyzes a suspicious card transaction using the RAG chain."""
    query = f"Analyze this card transaction for fraud: {transaction_details}"
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

def analyze_loan_application(application_details):
    """Analyzes a suspicious loan application using the RAG chain."""
    query = f"Analyze this loan application for fraud: {application_details}"
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

def analyze_authentication_event(auth_event_details):
    """Analyzes a suspicious authentication event using the RAG chain."""
    query = f"Analyze this authentication event for fraud: {auth_event_details}"
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

def analyze_payment(payment_details):
    """Analyzes a suspicious payment using the RAG chain."""
    query = f"Analyze this payment for fraud: {payment_details}"
    response = retrieval_chain.invoke({"input": query})
    return response["answer"]

# --- 8. Example usage of the RAG system ---
if __name__ == "__main__":
    print("--- Analyzing a suspicious card transaction ---")
    card_transaction = "A customer named John Doe in the US made a $6,000 purchase from a vendor in Spain. The customer has never made an international purchase before."
    response = analyze_card_transaction(card_transaction)
    print(response)

    print("\n--- Analyzing a suspicious loan application ---")
    loan_application = "A loan applicant's submitted documents show a residential address in Ohio, but the employment history shows a company based solely in California."
    response = analyze_loan_application(loan_application)
    print(response)

    print("\n--- Analyzing a suspicious authentication event ---")
    auth_event = "A customer's account received 5 failed login attempts in under a minute, all from a new, unrecognized IP address."
    response = analyze_authentication_event(auth_event)
    print(response)

    print("\n--- Analyzing a suspicious large payment ---")
    payment_event = "A customer initiated a $30,000 wire transfer to a new beneficiary. The customer's typical transfer amount is less than $1,000."
    response = analyze_payment(payment_event)
    print(response)

