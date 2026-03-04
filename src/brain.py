import os
from typing import List, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()

# 1. Define what our Graph remembers
class GraphState(TypedDict):
    question: str
    documents: List[str]
    generation: str

# 2. Setup the "Tools"
# Make sure this path matches where your ingestion.py created the folder
vector_db = Chroma(persist_directory="vector_db/", embedding_function=OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
model = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define the Actions (Nodes)
def retrieve(state: GraphState):
    print("--- STEP: RETRIEVING DOCS ---")
    question = state["question"]
    docs = retriever.invoke(question)
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
        
        content = f"CONTENT: {d.page_content}\nSOURCE: {source_file} (Page {page_num})\n"
        context_list.append(content)
    
    context = "\n\n".join(context_list)
    
    # 2. Update the prompt to demand specific citations
    prompt = f"""You are an expert medical assistant for Stroke RAG. 
    Answer the question using ONLY the context below. 
    
    CRITICAL INSTRUCTION: You must cite the specific document name and page number 
    for every claim you make (e.g., "Treatment X is recommended [stroke_prevention_2024.pdf, Page 12]").
    
    If the answer isn't in the context, say you don't know. 
    
    Question: {question}
    Context: {context}
    
    Answer:"""
    
    response = model.invoke(prompt)
    return {"generation": response.content, "question": question}

# 4. Connect the Dots
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

stroke_rag_app = workflow.compile()

if __name__ == "__main__":
    test_input = {"question": "What is the recommended treatment for high blood pressure in stroke prevention?"}
    result = stroke_rag_app.invoke(test_input)
    print("\nFINAL ANSWER:\n", result["generation"])