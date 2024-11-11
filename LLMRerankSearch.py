import torch

def read_results(file_path):
    query_results = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            query_id = parts[0]
            doc_id = parts[2]

            if query_id not in query_results:
                query_results[query_id] = []

            query_results[query_id].append(doc_id)

    return query_results


def rerank_documents(query, documents, model, tokenizer):
    scores = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for doc_id in documents:
        # this is the first prompt
        prompt = f"Query: {query}\nDocument: {doc_id}\nRank the relevance of this document from 1 (least relevant) to 10 (most relevant):"

        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        relevance_score = logits[0, -1].item()  # Use the logit of the last token

        scores.append((doc_id, relevance_score))

    # sort docs
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_docs

def write_ranked_results(query_results, output_file):
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, _ in documents:
                file.write(f"{query_id} Q0 {doc_id} {rank} 0 my_reranked_system\n")
                rank += 1