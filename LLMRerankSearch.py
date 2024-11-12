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

def rerank_documents(query_id, query_text, documents, model, tokenizer):
    scores = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_message = "You are a document ranking assistant who helps rank the relevance of documents based on a given query."

    for doc_id in documents:
        prompt = f"Query: {query_text}\nDocument: {doc_id}\nRank the relevance of this document from 1 (least relevant) to 10 (most relevant):"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512,
                           return_attention_mask=True).to(model.device)

        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, attention_mask=attention_mask, max_new_tokens=50,
                                     temperature=0.2, top_p=0.9, do_sample=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        relevance_score = compute_relevance_score(generated_text)

        scores.append((doc_id, relevance_score))

    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_docs

def compute_relevance_score(generated_text):
    return len(generated_text)

def write_ranked_results(query_results, output_file):
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, score in documents:
                file.write(f"{query_id} Q0 {doc_id} {rank} {score} my_reranked_system\n")
                rank += 1