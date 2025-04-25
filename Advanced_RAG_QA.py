import os
os.environ['USER_AGENT'] = 'myagent'
import time
import openai
import shutil
import nest_asyncio
from dotenv import load_dotenv
import pickle
import pandas as pd
import streamlit as st
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from trulens_eval import TruLlama
from trulens.dashboard.display import get_feedback_result
import numpy as np
from trulens.apps.llamaindex import TruLlama
from trulens.core import Feedback
import pandas as pd
from llama_index.core.tools import QueryEngineTool
from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

from llama_index.llms.openai import OpenAI
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from copy import deepcopy
from llama_index.core.schema import TextNode
load_dotenv()
LLAMACLOUD_API_KEY = os.getenv('LLAMACLOUD_API_KEY')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMACLOUD_API_KEY
# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


st.title("Advanced RAG System")
if "initialized" not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:

    """Instantiating the Embedding Model"""
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini")
    """Instantiating the LLM"""
    Settings.llm = llm
    Settings.embed_model = embed_model
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1024, chunk_overlap=20
    )
    st.write("Loading the files")
    # to convert pickle to documents form   
    pickle_file_path = "financial_documents.pkl"
    with open(pickle_file_path, "rb") as f:
        p_documents = pickle.load(f)
    print(f"Loaded {len(p_documents)} paged of document!")

    def get_page_nodes(docs):
        nodes = []
        for doc in docs:
            # Split the text into chunks
            doc_chunks = text_splitter.split_text(doc.text)
            for doc_chunk in doc_chunks:
                node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
                )
            nodes.append(node)
        return nodes
    
    page_nodes = get_page_nodes(p_documents)
    print(f"Converted to {len(page_nodes)} page nodes!")

    st.write("Creating Vector Index and Summary Index")
    ######## Summary and Vector Index ########
    from llama_index.core import SummaryIndex, VectorStoreIndex
    summary_index = SummaryIndex(page_nodes)
    vector_index = VectorStoreIndex(page_nodes)
    st.write("Creating Vector and Summary Engine")
    # Query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    vector_query_engine = vector_index.as_query_engine(similarity_top_k=5)
    # Query tool


    from trulens.providers.openai import OpenAI
    provider = OpenAI()
    context = TruLlama.select_context(vector_query_engine)

    # Define a groundedness feedback function
    f_groundedness = (
        Feedback(
         provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(context.collect())  # collect context chunks into a list
        .on_output()
    )

    # Question/answer relevance between overall question and answer.
    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()
    # Question/statement relevance between question and each context chunk.
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        )
        .on_input()
        .on(context)
        .aggregate(lambda x: np.mean(x[x > 0.0]) if np.any(x > 0.0) else 0)
    )
    
    st.write("Creating Vector tool and Summary Tool")
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
        "Useful for summarization questions related to Indian budget 2024-25 and Indian Economic Survey 2024-25."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context related to Indian Budget 2024-25 and Indian Economic Survey 2024-25"
        ),
    )
    # Define Router Query Engine
    adv_query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=True
    )
    #  FallBAck selector to deal with no selection made by LLMSelector Case

    class FallbackSelector(LLMSingleSelector):
        """Custom Selector to handle cases where LLM fails to decide."""

        def select(self, query_bundle, metadatas):
            try:
                # Attempt to select a tool
                selection = super().select(query_bundle, metadatas)

                if not selection.inds:
                    raise ValueError("No selection made by LLM.")

                return selection

            except ValueError as e:
           
                # Ensure the fallback selection object has `ind`
                return type("FallbackSelection", (object,), {
                    "ind": 1,      # Single fallback index for compatibility
                    "inds": [1],   # Keep it as a list for consistency
                    "reason": "Selecting Vector Engine ."
                })()

    # Use the custom selector
    st.write("Instantaiting Advanced Query Engine ")

    adv_query_engine = RouterQueryEngine(
        selector=FallbackSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,  # index=0
            vector_tool,   # index=1 (default fallback)
        ],
        verbose=True
    )


    st.write("Instantiating Evaluation ")
    from trulens.core import TruSession
    adv_session = TruSession()
    adv_session.reset_database()

    from trulens_eval import TruLlama
    al = TruLlama(adv_query_engine)

    tru_query_engine_recorder = TruLlama(
        adv_query_engine,
        app_name="LlamaIndex_App_advanced",
        app_version="advanced",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
    )
    # Store objects in session state
    st.session_state.embed_model=embed_model
    st.session_state.llm=llm
    st.session_state.p_documents=p_documents
    st.session_state.page_nodes=page_nodes
    st.session_state.summary_index=summary_index
    st.session_state.vector_index=vector_index
    st.session_state.summary_query_engine = summary_query_engine
    st.session_state.vector_query_engine = vector_query_engine
    st.session_state.vector_tool=vector_tool
    st.session_state.summary_tool=summary_tool
    st.session_state.adv_query_engine = adv_query_engine
    st.session_state.tru_query_engine_recorder = tru_query_engine_recorder
    st.session_state.initialized = True  # Mark as initialized

    st.write("Initialization complete! ✅")


def run_adv_workflow(question):
    print("Inside run advanced workflow")
    tru_query_engine_recorder=st.session_state.tru_query_engine_recorder
    adv_query_engine=st.session_state.adv_query_engine
    with tru_query_engine_recorder as recording:
        print("Query: ",question)
        response=adv_query_engine.query(question)
    print(str(response))
    st.write(str(response))
    # Extract selector result
    selector_result = response.metadata["selector_result"]
    # Get the selected engine index
    #selected_engine_index = selector_result.selections[0].index  # Extracts `index=1`
    if hasattr(selector_result, "selections"):
        #selected_engine_index = selector_result.selections[0]["index"]
        selected_engine_index = selector_result.selections[0].index 
    else:
        selected_engine_index = 1

    # Run evaluation **only** for the vector engine (assuming vector_tool is engine 1)
    if selected_engine_index == 1:
        print("✅ Selected Engine Index:", selected_engine_index)
        st.write("✅ Selected Vector Store Index ")
        print("Running TruLens evaluation for vector-based query...")
        st.write("Running TruLens evaluation for vector-based query...")
        last_record = recording.records[-1]
        answer_relevance = get_feedback_result(last_record, "Answer Relevance")
        context_relevance = get_feedback_result(last_record, "Context Relevance")
        groundedness = get_feedback_result(last_record, "Groundedness")

        # Define file path
        output_file = "feedback_results.xlsx"

        # Write to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            for df, sheet_name in zip([answer_relevance, context_relevance, groundedness],["Answer Relevance", "Context Relevance", "Groundedness"]):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                # Access the worksheet
                worksheet = writer.sheets[sheet_name]
                workbook = writer.book
                # Create a cell format for text wrapping
                wrap_format = workbook.add_format({"text_wrap": True, "align": "left", "valign": "top"})
            # Auto-adjust column width and apply text wrapping
                for i, col in enumerate(df.columns):
                    max_width = max(df[col].astype(str).apply(len).max(), len(col)) + 2  # Dynamic column width
                    worksheet.set_column(i, i, min(max_width, 50), wrap_format)  # Max width 50 to prevent overflow

        print(f"✅ Feedback results saved to {output_file}")
        st.write(f"✅ Feedback results saved to {output_file}")
        sheets_dict = pd.read_excel(output_file, sheet_name=None, engine="openpyxl")
        # Display all sheets dynamically
        st.write("Evaluation Scores:")
        for sheet_name, df in sheets_dict.items():
            st.write(f"**Metric Name:** `{sheet_name}`")
            st.dataframe(df)  # Display each sheet

    else:
        print("✅ Selected Engine Index:", selected_engine_index)
        st.write("✅ Selected Summary Index ")
        print("The above is the response to summary-based query.")
        st.write("The above is the response to summary-based query.")


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

# Load the college logo


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
def load_questions(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

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
        run_adv_workflow(user_question)
    else:
        st.warning("Please enter a question.")
# Sidebar for additional info

st.sidebar.title("About")
st.sidebar.info("This is an advanced RAG system designed for financial document analysis.")
questions_text = load_questions("questions.txt")
st.sidebar.text_area("Sample Questions", questions_text, height=200)







