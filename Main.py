import sys
import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from LLMRerankSearch import rerank_documents_with_qg, read_results, write_ranked_results
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Ensure the environment variables are set to store models on netstore (IMPORTANT!)
os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'
os.environ['HF_HOME'] = '/mnt/netstore1_home/mandy.ho/HF'

# Your Hugging Face access token (replace with your actual token)
hf_token = "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"  # Replace with your Hugging Face access token

# Specify the model ID for Meta-Llama 3.1
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model and tokenizer from Hugging Face
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # Use FP16 for efficient memory usage
    device_map="auto",  # Let Accelerate manage the model's device placement
    use_auth_token=hf_token
)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

# Create the text-generation pipeline
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, model_kwargs={"torch_dtype": torch.float16})

# Set pad token ID to match the tokenizer's pad token
llm_pipeline.model.generation_config.pad_token_id = tokenizer.pad_token_id

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

    # Print memory allocation for debugging
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    print(f"Topics 1 Results (Total Topics: {len(topics_1_results)}):")
    for topic_id, doc_ids in topics_1_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    print(f"Topics 2 Results (Total Topics: {len(topics_2_results)}):")
    for topic_id, doc_ids in topics_2_results.items():
        print(f"Topic {topic_id}: {len(doc_ids)} documents")

    # Perform reranking for topics_1 using Query Generation method
    reranked_results_1 = {}
    with tqdm(total=len(topics_1_results), desc="Reranking Topics 1") as pbar:
        reranked_results_1 = rerank_documents_with_qg(topics_1_results, topics_1, answers, llm_pipeline)
        pbar.update(len(topics_1_results))

    # Save the reranked results for topics 1
    write_ranked_results(reranked_results_1, "reranked_results_1_qg.tsv")

    # Perform reranking for topics_2 using Query Generation method
    reranked_results_2 = {}
    with tqdm(total=len(topics_2_results), desc="Reranking Topics 2") as pbar:
        reranked_results_2 = rerank_documents_with_qg(topics_2_results, topics_2, answers, llm_pipeline)
        pbar.update(len(topics_2_results))

    # Save the reranked results for topics 2
    write_ranked_results(reranked_results_2, "reranked_results_2_qg.tsv")
