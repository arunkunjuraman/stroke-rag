import os
import pandas as pd
from datasets import Dataset
from brain import stroke_rag_app # Import your compiled graph
from dotenv import load_dotenv
from ragas.metrics.collections import (
    ContextPrecision,
    Faithfulness,
    AnswerRelevancy,
    ContextRecall
)
from openai import AsyncOpenAI
from ragas.llms import llm_factory
#from ragas.embeddings import OpenAIEmbeddings
from ragas.embeddings.base import embedding_factory
import asyncio
import json
from pathlib import Path
from ragas.run_config import RunConfig


def load_golden_dataset(file_path: str):
    """
    Loads the evaluation dataset from a JSON file.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Evaluation set not found at {file_path}")
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    print(f"--- LOADED {len(data)} TEST CASES ---")
    return data

async def run_evaluation():
    # 1. Load your Golden Dataset
    #test_cases = [
     #   {"question": "What is the Class 1 recommendation for BP targets?", "ground_truth": "<130/80 mm Hg"}        
    #]
    test_cases = load_golden_dataset("tests/eval_set.json")

    #There is one key that is different in the faithfulness.score function: retrieved_contexts
    #Answer relevancy and the rest doesn't have retrieved_contexts. If passed, it will error out.
    #We are creating two different lists to account for that difference. Not sure at this point if there is 
    #a better more elegant way to handle this difference.
    results = []
    context_recall_results = []
    answer_relevancy_results = [] 
    print("--- STARTING EVALUATION RUN ---")
    api_key = os.getenv("OPENAI_API_KEY")
    # Initialize the Judge LLM
    client = AsyncOpenAI(api_key=api_key)
    evaluator_llm = llm_factory('gpt-4o', client=client)
    embeddings = embedding_factory("openai", model="text-embedding-ada-002", client=client)

    # 3. Initialize the metric objects
    # We pass the LLM into each metric so it knows how to grade
    faithfulness = Faithfulness(llm=evaluator_llm)
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm,embeddings=embeddings)
    context_recall = ContextRecall(llm=evaluator_llm)
    context_precision = ContextPrecision(llm=evaluator_llm)

    
    scores = {}

    for i, case in enumerate(test_cases):
        # Run the RAG Graph
        response = stroke_rag_app.invoke({"question": case['question']})                        
        
        faithfulness_score = await faithfulness.ascore(
        user_input = response['question'],
        response = response["generation"],        
        retrieved_contexts = [doc.page_content for doc in response["documents"]],
        )

        answer_relevancy_score = await answer_relevancy.ascore(
        user_input = response['question'],
        response = response["generation"]
        )

        context_recall_score = await context_recall.ascore(
        user_input = response['question'],        
        retrieved_contexts = [doc.page_content for doc in response["documents"]],
        reference = case["ground_truth"],#the reference answer
        )

        context_precision_score = await context_precision.ascore(
        user_input = response['question'],
        reference = case["ground_truth"],        
        retrieved_contexts = [doc.page_content for doc in response["documents"]]
        )

        scores[i] = {
        "faithfulness": faithfulness_score.value,
        "recall": context_recall_score.value,
        "relevancy": answer_relevancy_score.value,
        "precision": context_precision_score.value
    }

    # 2. Convert to RAGAS Dataset
    #dataset = Dataset.from_list(results)
    #dataset = Dataset.from_list(results)
    #context_recall_dataset = Dataset.from_list(context_recall_results)
    #answer_relevancy_dataset = Dataset.from_list = (answer_relevancy_results)    
    

    # 4. Export Report
    df = pd.DataFrame.from_dict(scores, orient="index")
    df.index.name = "case"
    df.to_csv("evaluation_report.csv")

    print("\n--- EVALUATION COMPLETE ---")
    print(f"Faithfulness Score: {faithfulness_score.value}")
    print(f"Answer Relevancy Score: {answer_relevancy_score.value}")
    print(f"Context Recall Score: {context_recall_score.value}")
    print(f"Context Precision Score: {context_precision_score.value}")    

if __name__ == "__main__":
    asyncio.run(run_evaluation())