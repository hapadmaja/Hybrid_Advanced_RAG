# hybrid_15

import os
os.environ['USER_AGENT'] = 'myagent'

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import openai

import time
import shutil
import time
from groq import Groq
from langchain_groq import ChatGroq
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
st.image("Anna_univ.png", width=150)  
st.title("Hybrid RAG System")
#"""Instantiate the Embedding Model"""
def log_message(message):
    st.markdown(
        f"""
        <div style="
            padding: 10px; 
            border-radius: 5px; 
            background-color: #e6f2ff; /* Light Blue */
            border-left: 5px solid #007acc; /* Darker Blue */
            margin: 10px 0;
        ">
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )

if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    openai.api_key = os.getenv("OPENAI_API_KEY") 
    embed_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key="YOUR_OPEN_AI_KEY"  # Replace with your actual API key
    )
    """Instantiate LLM"""
    
    #from google.colab import userdata
    # The GROQ_API_KEY must be added to the Secrets scetion of COLAB
    llm = ChatGroq(temperature=0,
                        model_name="Llama3-70b-8192",
                        api_key="YOUR_LLAMA_KEY",)
    pure_llm = ChatGroq(temperature=0.3,
                        model_name="Llama3-70b-8192",
                        api_key="YOUR_LLAMA_KEY",)


    #"""Download the data"""
    # Path to the folder containing documents
    folder_path = "./economic_survey"  # Change this to your directory path

    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    # Load all PDFs
    all_documents = []

    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        
        # Load PDF using PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Append extracted documents
        all_documents.extend(documents)


    # Flatten and debug the document list
    docs_list = [doc for doc in all_documents]
    print(f"len of documents: {len(docs_list)}")

    #"""Chunk the Documents to be in sync with the context window of the LLM"""

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=20
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"length of document chunks generated :{len(doc_splits)}")

    print("Starting chroma vector Embedding")
    start = time.time()


    CHROMA_DB_DIR = "chromadb"

    # Clear directory at the start of the program (Only Once)
    if not st.session_state.get("cleared_db", False):
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)  # Delete the directory
            #st.write("🗑️ ChromaDB directory cleared!")
            log_message("🗑️ ChromaDB directory cleared!")
        st.session_state["cleared_db"] = True  # Mark as cleared

    def load_vectorstore(doc_splits, embed_model):
        return Chroma.from_documents(
            documents=doc_splits,
            embedding=embed_model,
            collection_name="local-rag",
            persist_directory=CHROMA_DB_DIR
        )  

    # Ensure documents are processed only once
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore(doc_splits, embed_model)
        
    #Load the documents to vectorstore
    vectorstore = st.session_state.vectorstore

    end = time.time()
    print("Ending chroma vector Embedding")
    print(f"The time required for chroma db:{end - start}")

    #st.write("Vectorstore loaded successfully!")
    log_message("Vectorstore loaded successfully!")
    print("Vectorstore loaded successfully!")

    #Instiated the retriever

    retriever = vectorstore.as_retriever(search_kwargs={"k":5})
    #st.write("retreiver Instantiated")
    log_message("retreiver Instantiated")
    print("retreiver Instantiated")


    """Implement the Router"""

    

    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
        user question to a vectorstore or web search. Use the vectorstore for questions tax regimes, tax policies, financial rules , budget themes, indirect taxes, direct taxes, Interim budget. You do not need to be stringent with the keywords
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
        no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    start = time.time()
    question_router = prompt | llm | JsonOutputParser()

    """Implement the Generate Chain"""

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Context: {context}
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "document"],
    )
    pure_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
        Use your knowledge to answer the question. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    start = time.time()
    rag_chain = prompt | llm | StrOutputParser()
    # added pure rag chain
    pure_rag_chain = pure_prompt | pure_llm | StrOutputParser()
    #Implement the Retrieval Grader
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )
    start = time.time()
    retrieval_grader = prompt | llm | JsonOutputParser()
    #Implement the hallucination grader
    # Prompt
    generation = []
    prompt = PromptTemplate(
        template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        keys 'score' and 'reason'. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here are the facts: 
        \n ------- \n
        {documents}
        \n ------- \n
        Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "documents"],
    )
    start = time.time()
    hallucination_grader = prompt | llm | JsonOutputParser()


    #Implement the Answer Grader
    # Prompt
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
        useful to resolve a question. Provide the binary score as a JSON with keys 'score' and 'reason'
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation}
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["generation", "question"],
    )
    start = time.time()
    answer_grader = prompt | llm | JsonOutputParser()

    transform_query_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        You are a query transformation agent. Your task is to rewrite a given question to make it clearer, more precise, 
        and optimized for web search. Ensure that the transformed question retains the original intent but is structured 
        in a way that maximizes the retrieval of relevant and high-quality information.  
        
        Provide the transformed question as a JSON with keys 'transformed_query' and 'reason' explaining why the transformation 
        was necessary.  
        <|eot_id|><|start_header_id|>user<|end_header_id|>  
        Original Query:  
        \n ------- \n  
        {question}  
        \n ------- \n  
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    start = time.time()
    question_rewriter = transform_query_prompt | llm | JsonOutputParser()

    st.session_state.retriever=retriever
    st.session_state.rag_chain=rag_chain
    st.session_state.pure_rag_chain=pure_rag_chain
    st.session_state.hallucination_grader=hallucination_grader
    st.session_state.answer_grader=answer_grader
    st.session_state.retrieval_grader=retrieval_grader
    st.session_state.question_rewriter=question_rewriter
    st.session_state.question_router=question_router
    st.session_state.initialized = True  # Mark as initialized
    st.write("Initialization complete! ✅")

retriever=st.session_state.retriever
rag_chain=st.session_state.rag_chain
pure_rag_chain=st.session_state.pure_rag_chain
hallucination_grader=st.session_state.hallucination_grader
answer_grader=st.session_state.answer_grader
retrieval_grader= st.session_state.retrieval_grader
question_rewriter=st.session_state.question_rewriter
question_router=st.session_state.question_router
#Implement Web Search tool
import os
from langchain_community.tools.tavily_search import TavilySearchResults
os.environ['TAVILY_API_KEY'] = "YOUR_TAVILY_KEY"
web_search_tool = TavilySearchResults(k=3)

from typing_extensions import TypedDict
from typing import List

### State

class GraphState(TypedDict):
    question : str
    generation : str
    web_search : str
    documents : List[str]
    websearch_retries :int
    generate_retries:int
    pure_response : str
    def __init__(self, question: str = "", generation: str = "", web_search: str = "", documents: List[str] = None):
        self.question = question
        self.generation = generation
        self.web_search = web_search
        self.documents = documents if documents is not None else []
        self.websearch_retries = 1
        self.generate_retries = 1

    def reset_retries(self):
        """Resets retry counters when a new query starts."""
        self.websearch_retries = 1
        self.generate_retries = 1
    
    

#Define the Nodes
from langchain.schema import Document
def start_new_query(state: dict) -> dict:
    state["websearch_retries"] = 1
    state["generate_retries"] = 1
    return state

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    #st.write("---RETRIEVING RELEVANT CONTEXT---")
    log_message("---RETRIEVING RELEVANT CONTEXT---")
    question = state["question"]
    pure_response=state["pure_response"]
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question, "pure_response":pure_response}
#
def pure_retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    #print("---RETRIEVE---")
    # actally retriveing docs
    print("---PASSING QUESTION TO LLM---")
    #st.write("---PASSING QUESTION TO LLM---")
    log_message("---PASSING QUESTION TO LLM---")
    question = state["question"]
    print("---PARALLELY RETRIEVING CONETXT---")
    log_message("---PARALLELY RETRIEVING RELEVANT CONTEXT---")
    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE IN RAG ---")
    #st.write("---GENERATE---")
    log_message("---GENERATE IN RAG---")
    question = state["question"]
    documents = state["documents"]
    pure_response= state["pure_response"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "pure_response":pure_response}

def pure_generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATION BY ONLY LLM WITHOUT AUGMENTATION---")
    #st.write("---GENERATE---")
    log_message("---GENERATE BY LLM WITHOUT AUGMENTATION---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    pure_generation = pure_rag_chain.invoke({"question": question})
    return {"documents": documents,"question": question, "generation": pure_generation,"pure_response": pure_generation}


def pure_grade_hallucination_answer(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK GROUNDEDNESS---")
    #st.write("---CHECK GROUNDEDNESS---")
    log_message("---CHECK GROUNDEDNESS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["pure_response"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']
    reason= score['reason']
    print("Response of LLM: ",generation)
    #print("Context :",documents)
    print("Score returned by grader: ",grade)
    print("Reason provided by grader: ",reason)
    print("\n")
    # score indicates yes: answer grounded in docs and no indicates the response is NOT grounded in docs
    # Check hallucination
    if grade == "yes":
        print("---DECISION: PURE GENERATION IS NOT HALLUCINATED---")
        #st.write("---DECISION: PURE GENERATION IS NOT HALLUCINATED---")
        log_message("---DECISION: PURE GENERATION IS NOT HALLUCINATED---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        #st.write("---GRADE GENERATION vs QUESTION---")
        log_message("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        reason= score['reason']
        print("Response of LLM: ",generation)
        print("Score returned by grader: ",grade)
        print("Reason provided by grader: ",reason)
        print("\n")
        if grade == "yes":
            print("---DECISION: PURE GENERATION ADDRESSES QUESTION---")
            #st.write("---DECISION: PURE GENERATION ADDRESSES QUESTION---")
            log_message("---DECISION: PURE GENERATION ADDRESSES QUESTION---")
            return "useful_p"
        else:
            print("---DECISION: PURE GENERATION DOES NOT ADDRESS QUESTION---")
            #st.write("---DECISION: PURE GENERATION DOES NOT ADDRESS QUESTION---")
            log_message("---DECISION: PURE GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful_p"
    else:
        pprint("---DECISION: PURE GENERATION IS HALLUCINATED, RE-TRY WITH RAG---")
        #st.write("---DECISION: PURE GENERATION IS HALLUCINATED, RE-TRY WITH RAG---")
        log_message("---DECISION: PURE GENERATION IS HALLUCINATED, RE-TRY WITH RAG---")
        return "not supported_p"
    
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    #st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    log_message("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    pure_response= state["pure_response"]
    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search,"pure_response": pure_response}
#
def pure_grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    #st.write("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    log_message("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    #print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    #st.write("---WEB SEARCH---")
    log_message("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])


    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
def pure_web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    #print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])


    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", [])
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    trans_query= better_question['transformed_query']
    return {"documents": documents, "question": trans_query}
def pure_transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    #print("---PURE--TRANSFORM QUERY---")
    question = state["question"]
    documents = state.get("documents", [])
    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    trans_query= better_question['transformed_query']
    return {"documents": documents, "question": trans_query}
 

#Define Conditional Edges
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    #st.write("---ASSESS GRADED DOCUMENTS---")
    log_message("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes" and len(filtered_documents)==0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        #st.write("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        log_message("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        #return "websearch"
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        #st.write("---DECISION: GENERATE---")
        log_message("---DECISION: GENERATE---")
        return "generate"
def pure_decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    print("---ASSESS GRADED DOCUMENTS---")
    #st.write("---ASSESS GRADED DOCUMENTS---")
    log_message("---ASSESS GRADED DOCUMENTS---")
    #print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes" and len(filtered_documents)==0:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        log_message("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "pure_transform_query"
    else:
        # We have relevant documents, so generate answer
        #print("---DECISION: GENERATE---")
        print("---DECISION: PROCEEDING TO GROUNDEDNESS GRADING---")
        #st.write("---DECISION: GENERATE---")
        log_message("---DECISION: PROCEEDING TO GROUNDEDNESS GRADING---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK GROUNDEDNESS ---")
    #st.write("---CHECK GROUNDEDNESS ---")
    log_message("---CHECK GROUNDEDNESS ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']
    reason= score['reason']
    print("Response of system: ",generation)
    #print("Context :",documents)
    print("Score returned by grader: ",grade)
    print("Reason provided by grader: ",reason)
    print("\n")
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        #st.write("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        log_message("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        #st.write("---GRADE GENERATION vs QUESTION---")
        log_message("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        reason= score['reason']
        print("Response of LLM: ",generation)
        print("Score returned by grader: ",grade)
        print("Reason provided by grader: ",reason)
        print("\n")
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            #st.write("---DECISION: GENERATION ADDRESSES QUESTION---")
            log_message("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            if state.get("websearch_retries", 0) > 3:
                print("CAUTION : RETRY EXHAUSTED MAX LIMIT IS 3 \n The answer might not satisfy your query")
                log_message("CAUTION : RETRY EXHAUSTED MAX LIMIT IS 3 \n The answer might not satisfy your query")
                return "exhausted_websearch_retry"
            print("web search retry turn ",state.get("websearch_retries"))
            pprint("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            log_message("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
            
    else:
        if state.get("generate_retries", 0) > 3:
            print("CAUTION : RETRY EXHAUSTED \n The answer might be hallucinated, KINDLY check with the documents for clarity")
            log_message("CAUTION : RETRY EXHAUSTED \n The answer might be hallucinated, KINDLY check with the documents for clarity")
            return "exhausted_generate_retry"
        print("generate retry turn :",state.get("generate_retries"))
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        log_message("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

#Add nodes
from langgraph.graph import END,START,StateGraph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search) # web search
workflow.add_node("pure_websearch", pure_web_search) # web search
workflow.add_node("transform_query", transform_query) # query transformation
workflow.add_node("pure_transform_query", pure_transform_query) # query transformation
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("pure_retrieve", pure_retrieve) # retrieve
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("pure_grade_documents", pure_grade_documents) # grade documents
workflow.add_node("generate", generate) # generate
workflow.add_node("pure_generate", pure_generate) # generate
workflow.add_node("start_new_query",start_new_query)


#Set the Entry Point and End Point
workflow.add_edge(START, "start_new_query")
workflow.add_edge("start_new_query", "pure_retrieve")  # Proceed normally
workflow.add_edge("pure_retrieve", "pure_grade_documents")
workflow.add_conditional_edges(
    "pure_grade_documents",
    pure_decide_to_generate,
    {
        "pure_transform_query": "pure_transform_query",
        "generate": "pure_generate",
    },
)
workflow.add_edge("pure_transform_query", "pure_websearch")
workflow.add_edge("pure_websearch", "pure_generate")
workflow.add_conditional_edges(
    "pure_generate",
    pure_grade_hallucination_answer,
    {
        "not supported_p": "generate",
        "useful_p": END,
        "not useful_p": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
        "exhausted_generate_retry":END,
        "exhausted_websearch_retry":END,
    },
)
workflow.add_edge("transform_query", "websearch")
workflow.add_edge("websearch", "generate")

#Compile the workflow
app = workflow.compile()

#Test the Workflow

from pprint import pprint


#Compile the workflow
app = workflow.compile()

#Test the Workflow
from pprint import pprint

from pprint import pprint

def boxed_response(response, title, color):
    st.markdown(
        f"""
        <div style="
            border: 2px solid {color}; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0px; 
            background-color: rgba(255, 255, 255, 0.1);
        ">
            <h4 style="color: {color};">{title}</h4>
            <p>{response}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def run_rag_workflow(question):
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            pprint(f"Finished running: {key}:")
    pprint(value["generation"])
    pure_response = value.get("pure_response", "No pure response available")
    rag_response = value.get("generation", "No RAG-based response available")

    if pure_response != rag_response:
        col1, col2 = st.columns(2)  # Create two side-by-side columns

        with col1:
            boxed_response(pure_response, "🔴 Pure LLM Response ❌", "red")

        with col2:
            boxed_response(rag_response, "🟢 RAG-Based Response ✅", "green")
    else:
        boxed_response(pure_response, "🟢 Pure LLM Response ✅", "green")
            

    

# Streamlit UI
# Set the folder path where PDFs are stored
FOLDER_PATH = "./economic_survey"  # Change this to your actual folder path

# Function to fetch PDF document names from the folder
def get_loaded_documents(folder_path):
    try:
        files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
        return files
    except Exception as e:
        return [f"Error accessing folder: {e}"]


# Custom CSS for styling
st.markdown(
    """
    <style>
        .title {
            color: #004aad;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .sidebar-text {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Sidebar to display document names
st.sidebar.title("📂 Documents Loaded")

# Fetch PDF document names
documents = get_loaded_documents(FOLDER_PATH)

if documents:
    for doc in documents:
        st.sidebar.markdown(f'<p class="sidebar-text">📄 {doc}</p>', unsafe_allow_html=True)
else:
    st.sidebar.warning("No PDFs found in the folder.")
user_question = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_question:
        run_rag_workflow(user_question)
    else:
        st.warning("Please enter a question.")
# Sidebar for additional info
st.sidebar.image("Anna_univ.png", width=100)
st.sidebar.title("About")
st.sidebar.info("This is an Hybrid RAG system designed for financial document analysis with self corrective mechanism")
