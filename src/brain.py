import os
from typing import List, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
import yaml

load_dotenv()

# 1. Define what our Graph remembers
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# 2. Setup the "Tools"
# Make sure this path matches where your ingestion.py created the folder
vector_db = Chroma(persist_directory="vector_db/", embedding_function=OpenAIEmbeddings())
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 20})
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Setup BM25 (Sparse) 
# Architect Note: BM25 usually needs the doc list. In a production 
# scale-up, you'd use a dedicated search engine like Elasticsearch.
all_docs = vector_db.get() # Retrieve documents from your local Chroma store
documents = [Document(page_content=text, metadata=meta) 
             for text, meta in zip(all_docs['documents'], all_docs['metadatas'])]

keyword_retriever = BM25Retriever.from_documents(documents)
keyword_retriever.k = 20

# 4. Create the Hybrid Ensemble
# We weigh them 50/50. You can tune this (e.g., 0.7 for Semantic if it's too literal)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, keyword_retriever],
    weights=[0.5, 0.5]
)
# 5. Define the Re-ranker (Precision)
# This model evaluates the Query + Chunk pair specifically
compressor = CohereRerank(model="rerank-english-v3.0", top_n=5)

# 6. Create the "Smart" Retriever
smart_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid_retriever
)
# 5. Define the Actions (Nodes)
def retrieve(state: GraphState):
    print("--- STEP: HYBRID RETRIEVING (DENSE + SPARSE) + COHERE RE-RANK ---")
    question = state["question"]
    #docs = hybrid_retriever.invoke(question)
    docs = smart_retriever.invoke(question) 

    return {"documents": docs, "question": question}

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

# 4. Connect the Dots
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

stroke_rag_app = workflow.compile()

if __name__ == "__main__":
    #test_input = {"question": "What is the recommended treatment for high blood pressure in stroke prevention?"}
    #test_input = {"question": "What are the specific Class I recommendations for antiplatelet therapy in the 2021 guidelines?"}
    test_input = {"question": "What is the Class 1 recommendation for BP targets?"}
    result = stroke_rag_app.invoke(test_input)
    print("\nFINAL ANSWER:\n", result["generation"])