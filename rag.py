from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from duckduckgo_search import DDGS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def init_llm():
    """Initialize the language model"""
    load_dotenv()
    os.environ['USER_AGENT'] = 'USER_AGENT'
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0
    )
    return llm

def process_pdfs(file_paths):
    """Process PDF files and create vector store"""
    # Load and process documents
    docs = [PyMuPDFLoader(file_path).load() for file_path in file_paths]
    docs_list = [item for sublist in docs for item in sublist]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Create vector store
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever(k=6)

def query_documents(question, retriever, llm):
    """Query the documents using RAG"""
    # Get relevant documents
    docs = retriever.invoke(question)
    docs_string = "".join(doc.page_content for doc in docs)
    
    # Create prompt template
    template = PromptTemplate.from_template("""
    You are a helpful assistant who is good at analyzing source information and answering questions.       
    Use the following source documents to answer the user's questions.      
    If you don't know the answer, just say 'I don't know'.       
    If the answer is not directly in the documents, just say 'I don't know'.
    Do not make an answer up.
    Use three sentences maximum and keep the answer concise.

    Documents:
    {docs_string}
    User question: 
    {question}
    Answer:
    """.strip())
    
    chain = template | llm
    response = chain.invoke({"question": question, "docs_string": docs_string})
    return response.content

def web_search(question, llm):
    """Perform web search when document search fails"""
    duckduckgo = DDGS(timeout=20)
    
    # Convert question to search query
    template1 = PromptTemplate.from_template(
        """You are an analyst tasked with converting questions to a form more suitable for search engine queries. 
        Convert the following to a search engine query: {question}. Query:""".strip()
    )
    chain1 = template1 | llm
    search_query = chain1.invoke({"question": question})
    
    # Perform search
    search_result = duckduckgo.text(search_query.content, max_results=7)
    search_results = "\n\n".join([
        "{_index}. {_title}\n{_body}\nSource: {_href}".format(
            _index=i + 1,
            _title=result["title"],
            _body=result["body"],
            _href=result["href"],
        )
        for i, result in enumerate(search_result)
    ])
    
    # Process search results
    template2 = PromptTemplate.from_template("""
    You are a helpful search assistant. Be polite and informative. Always try to not be wrong.
    Only use the search results below to answer. Answer the user's question based on the search results below.
    Use three sentences maximum and keep the answer concise.
    
    Search results:
    {search_results}
    User question: 
    {question}
    Answer:
    """.strip())
    
    chain2 = template2 | llm
    response = chain2.invoke({"question": question, "search_results": search_results})
    return response.content

def get_answer(question, retriever, llm):
    """Main function to get answers from documents or web search"""
    # First try documents
    answer = query_documents(question, retriever, llm)
    
    # If no answer found, try web search
    if answer == "I don't know.":
        web_answer = web_search(question, llm)
        return web_answer, True
    
    return answer, False