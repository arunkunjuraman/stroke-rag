import os
from typing import List, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import yaml
from cache import cache
from logger import logger



load_dotenv()

# 1. Define what our Graph remembers
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_cached: bool


# 2. Setup the "Tools"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_db = Chroma(
    collection_name="stroke_rag_parents", 
    persist_directory="vector_db_parent/", 
    embedding_function=embeddings
)

# Parent Document Retrieval setup
fs = LocalFileStore("./parent_store")
store = create_kv_docstore(fs)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

parent_retriever = ParentDocumentRetriever(
    vectorstore=vector_db,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 40} # Increased from 20 to widen the candidate pool
)

model = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Setup BM25 (Sparse) 
# To improve recall and consistency with our Parent retrieval, 
# we also index the larger context (Parent) documents in our keyword search.
parent_keys = list(fs.yield_keys())
parent_docs = store.mget(parent_keys)

keyword_retriever = BM25Retriever.from_documents(parent_docs)
keyword_retriever.k = 40 # Increased to match vector retriever depth

# 4. Create the Hybrid Ensemble
# Combine the precision of Parent-Child vector search with Keyword search (Semantic + Keyword)
hybrid_retriever = EnsembleRetriever(
    retrievers=[parent_retriever, keyword_retriever],
    weights=[0.7, 0.3] # Increased Vector weight to boost semantic accuracy
)
# 5. Define the Re-ranker (Precision)
# This model evaluates the Query + Chunk pair specifically
# Increasing top_n to 5 to provide more context to the LLM for complex comparative questions
compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)

# 6. Create the "Smart" Retriever
smart_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid_retriever
)
# 7. Multi-Query Expansion for Recall
query_expansion_prompt = ChatPromptTemplate.from_template(
    "You are an AI specialized in clinical guidelines for stroke. "
    "Given the question below, generate 3 distinct search variations to maximize retrieval recall:\n"
    "1. A direct clinical version using medical terminology.\n"
    "2. A keyword-dense version focusing on the year and specific guidelines (e.g., 2024, AHA/ASA).\n"
    "3. A summary-level version to identify broader recommendation sections.\n\n"
    "Original question: {question}\n"
    "Output only the 3 variations, one per line."
)
search_generator = query_expansion_prompt | model | StrOutputParser()

# 5. Define the Actions (Nodes)
def retrieve(state: GraphState):
    print("--- STEP: MULTI-QUERY EXPANSION + HYBRID RETRIEVING + RE-RANK ---")
    question = state["question"]
    
    # Expand query for better focus on specific guideline semantics
    variations = search_generator.invoke({"question": question}).split("\n")
    queries = [question] + [v.strip() for v in variations if v.strip()]
    
    all_docs = []
    # Collect candidates from all variations (Ensemble handles duplicates via ranking logic usually, 
    # but for simplicity we manually deduplicate if needed, though most retrievers handle it)
    for q in queries[:3]: # Limit to original + top 2 variants for speed
        docs = smart_retriever.invoke(q) 
        all_docs.extend(docs)
    
    # Simple deduplication by content
    unique_docs = {d.page_content: d for d in all_docs}.values()
    
    return {"documents": list(unique_docs), "question": question}

def generate(state: GraphState):
    print("--- STEP: GENERATING PROFESSIONAL ANSWER ---")
    question = state["question"]
    docs = state["documents"]
    
    # 1. Format context with rich metadata (Filename and Page)
    context_list = []
    for i, d in enumerate(docs):
        # Extract just the filename from the full path
        source_file = os.path.basename(d.metadata.get('source', 'Unknown'))
        page_num = d.metadata.get('page', 'Unknown')
        
        # Format for the LLM
        content = f"CONTENT: {d.page_content}\nSOURCE: {source_file} (Page {page_num})\n"
        context_list.append(content)
    
    context = "\n\n".join(context_list)
        
    prompt = current_prompt_template.format(
        context=context, 
        question=state["question"]
    )
    response = model.invoke(prompt)
    return {"generation": response.content, "question": question}

def load_prompt(version_file):
    with open(f"prompts/{version_file}", 'r') as f:
        data = yaml.safe_load(f)
        return data['template']

current_prompt_template = load_prompt("v2_comparative.yaml")

def check_cache(state: GraphState):
    print("--- STEP: CHECKING CACHE ---")
    question = state["question"]
    cached_res = cache.get_response(question)
    if cached_res:
        return {
            "generation": cached_res["generation"], 
            "documents": cached_res["documents"], 
            "is_cached": True
        }
    return {"is_cached": False}

def save_cache(state: GraphState):
    if not state.get("is_cached", False):
        print("--- STEP: SAVING TO CACHE ---")
        cache.set_response(state["question"], state)
    return state

def log_results(state: GraphState):
    print("--- STEP: LOGGING INTERACTION ---")
    logger.log_interaction(state)
    return state



# 4. Connect the Dots
workflow = StateGraph(GraphState)
workflow.add_node("check_cache", check_cache)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("save_cache", save_cache)
workflow.add_node("log_results", log_results)

workflow.set_entry_point("check_cache")

def decide_to_retrieve(state: GraphState):
    if state.get("is_cached", False):
        return "log"
    else:
        return "retrieve"

workflow.add_conditional_edges(
    "check_cache",
    decide_to_retrieve,
    {
        "log": "log_results",
        "retrieve": "retrieve"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "save_cache")
workflow.add_edge("save_cache", "log_results")
workflow.add_edge("log_results", END)



stroke_rag_app = workflow.compile()

if __name__ == "__main__":
    #test_input = {"question": "What is the recommended treatment for high blood pressure in stroke prevention?"}
    #test_input = {"question": "What are the specific Class I recommendations for antiplatelet therapy in the 2021 guidelines?"}
    test_input = {"question": "What is the Class 1 recommendation for BP targets?"}
    result = stroke_rag_app.invoke(test_input)
    print("\nFINAL ANSWER:\n", result["generation"])