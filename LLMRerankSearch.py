import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TFIDFBaseSearch import TFIDFModel


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

def rerank_documents(query_id, query_text, documents, model, tokenizer, answers):
    scores = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_message = "You are a document ranking assistant who helps rank the relevance of documents based on a given query."

    # Retrieve the correct answer text for the current query
    answer_text = next((answer["Text"] for answer in answers if answer["Id"] == query_id), "")

    for doc in documents:  # Iterate over the document dictionaries
        doc_id = doc['Id']  # Get the document ID
        doc_text = get_document_text(doc_id, documents)  # Pass 'documents' to the function

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

        relevance_score = adjust_score_based_on_answer(doc_id, answer_text, doc_text, relevance_score)

        scores.append((doc_id, relevance_score))

    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_docs


def compute_relevance_score(generated_text):
    return len(generated_text)


def get_document_text(doc_id, documents):
    doc = next((doc for doc in documents if doc['Id'] == doc_id), None)

    if doc:
        doc_text = doc.get('Text', '')

        tfidf_model = TFIDFModel()
        cleaned_text = tfidf_model.clean_text(doc_text)

        return cleaned_text
    else:
        print(f"Document with ID {doc_id} not found!")
        return ""


def adjust_score_based_on_answer(doc_id, answer_text, doc_text, relevance_score):
    """Adjust the relevance score based on cosine similarity between document and answer text."""
    if not answer_text or not doc_text:
        return relevance_score

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    vectors = tfidf_vectorizer.fit_transform([doc_text, answer_text])

    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    if cosine_sim > 0.5:
        relevance_score += 2 * cosine_sim

    return relevance_score

def write_ranked_results(query_results, output_file):
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, score in documents:
                file.write(f"{query_id} Q0 {doc_id} {rank} {score} my_reranked_system\n")
                rank += 1