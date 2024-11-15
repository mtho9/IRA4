import sys
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLMRerankSearch import read_results, rerank_documents_with_qg, write_ranked_results

# Set environment variables for model cache and Hugging Face home
os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
os.environ['HF_HOME'] = '/mnt/netstore1_home/mandy.ho/HF'

# Model ID for Llama-3.2-1B-Instruct
model_id = "meta-llama/Llama-3.2-1B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Optional: Use bfloat16 for mixed precision
    device_map="auto",  # Automatically map the model to available GPUs
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Ensure the pad_token is set correctly (set to eos_token if None)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create the text-generation pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.float16},  # Use mixed precision
)

# Set pad_token_id for the model's generation config to avoid warnings
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