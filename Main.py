import sys
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Optional: Set the directory where the model is stored locally
local_model_path = '/home/mandy.ho/IRA4/Llama-3.2-1B-Instruct'  # This is where you downloaded the model

# Set the device (CUDA or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer from the local directory
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,  # Path to the local directory where the model was downloaded
    torch_dtype=torch.float16,  # You can use torch.bfloat16 as well, depending on your setup
    device_map="auto"  # Automatically map the model to available GPUs (if any)
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)  # Load tokenizer from the same local path

# Create the text-generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},  # Use float16 for mixed precision
)
# Set the pad token ID to match the tokenizer's pad token
llm_pipeline.model.generation_config.pad_token_id = tokenizer.pad_token_id
llm_pipeline.model.eval()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
        sys.exit(1)

    # Read input files
    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    with open(answers_file, 'r') as f:
        answers = json.load(f)
    with open(topics_1_file, 'r') as f:
        topics_1 = json.load(f)
    with open(topics_2_file, 'r') as f:
        topics_2 = json.load(f)

    # Read result files
    topics_1_results = read_results("bm25_1.tsv")
    topics_2_results = read_results("bm25_2.tsv")

    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())

    print(f"Topics 1 Results (Total Topics: {len(topics_1_results)}):")
    for topic_id, doc_ids in topics_1_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    print(f"Topics 2 Results (Total Topics: {len(topics_2_results)}):")
    for topic_id, doc_ids in topics_2_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    # Perform reranking for topics_1 using Query Generation method
    reranked_results_1 = rerank_documents_with_qg(topics_1_results, topics_1, answers, llm_pipeline, tokenizer, device)
    write_ranked_results(reranked_results_1, "reranked_results_1_qg.tsv")

    # Perform reranking for topics_2 using Query Generation method
    reranked_results_2 = rerank_documents_with_qg(topics_2_results, topics_2, answers, llm_pipeline, tokenizer, device)
    write_ranked_results(reranked_results_2, "reranked_results_2_qg.tsv")