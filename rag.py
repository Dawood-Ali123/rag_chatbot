from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
#embedding model
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore=Chroma(
    collection_name="chatbot_documents",
    embedding_function=embeddings,
    persist_directory="./chorma_db"
)
def add_pdf(file_path,file_name):
    loader=PyPDFLoader(file_path)
    documnets=loader.load()
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=splitter.split_documents(documnets)
    for chunk in chunks:
        chunk.metadata['source']=file_name
    vectorstore.add_documents(chunks)
    return len(chunks)
@tool
def search_uploaded_documents(query:str)->str:
    """
    Search the uploaded documnets for the relevant information
    use this tool when the user asks a question
    related to their documents
    """
    docs=vectorstore.similarity_search(
        query,
        k=4
    )
    if not docs:
        return "No relevant information found in the uploaded documents"
    results=[]
    for doc in docs:
        results.append(
            f"""
Source:{doc.metadata.get('source','unknown')}
content:{doc.page_content}
"""
        )
        return "\n\n---\n\n.join(results)"