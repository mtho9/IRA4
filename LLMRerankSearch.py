import torch
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
    "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't", '-', '.'
])

def clean_text(text):
    """Clean the input text by removing HTML, punctuation, and stop words."""
    text = remove_html(text)
    text = remove_punctuation(text).lower()
    text = remove_stop_words(text)
    return text

def remove_html(text):
    """Remove HTML tags from the text."""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ')

def remove_punctuation(text):
    """Remove punctuation from the text."""
    return re.sub(r'[^\w\s]', '', text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def read_results(file_path) -> dict:
    """Reads the results file and returns a dictionary with query IDs as keys and a list of document IDs as values."""
    topicdict = {}

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()

            topic_id = parts[0]
            doc_id = parts[2]

            if topic_id not in topicdict:
                topicdict[topic_id] = []  # if the topic ID is not a key, create an empty list for the associated docs

            topicdict[topic_id].append(doc_id)  # add doc ID to the list

    return topicdict  # returns dictionary mapping query IDs to list of doc IDs returned from tf-idf search


def rerank_documents(topicdict: dict, topics: list, answers: list, model, tokenizer) -> dict:
    reranked_results = {}

    for topic_id, doc_ids in topicdict.items():

        topic_parts = next((topic for topic in topics if topic['Id'] == topic_id), None)
        if not topic_parts:
            print(f"Topic ID {topic_id} not found in topics.")
            continue

        topic_title = topic_parts['Title']
        topic_body = topic_parts['Body']

        topic_text = f"{topic_title} {topic_body}"
        topic_terms = clean_text(topic_text).split()

        answer_text = next((answer['Text'] for answer in answers if answer['Id'] == topic_id), "")

        scores = {}

        for doc_id in doc_ids:
            doc_text = next((answer['Text'] for answer in answers if answer['Id'] == doc_id), "")

            prompt = f"Query: {topic_terms}\nDocument: {doc_id}\nRank the relevance of this document from 1 (least relevant) to 5 (most relevant):"

            system_message = "You are a document ranking assistant who helps rank the relevance of documents based on a given query."
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

            try:
                model_score = int(generated_text.strip())
            except ValueError:
                print(f"Error: Unable to convert model output to an integer: {generated_text}")
                model_score = 0

            relevance_score = adjust_score_based_on_answer(doc_id, answer_text, doc_text, model_score)

            scores[doc_id] = relevance_score

        reranked_results[topic_id] = {doc_id: score for doc_id, score in
                                      sorted(scores.items(), key=lambda x: x[1], reverse=True)}

    return reranked_results

def adjust_score_based_on_answer(doc_id, answer_text, doc_text, model_score):
    """Adjust the model's relevance score based on cosine similarity between document and answer text."""

    if not answer_text or not doc_text:
        return model_score

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    vectors = tfidf_vectorizer.fit_transform([doc_text, answer_text])

    cosine_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    if cosine_sim > 0.5:
        model_score += 2 * cosine_sim

    return model_score

def write_ranked_results(query_results, output_file):
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, score in documents.items():
                file.write(f"{query_id} Q0 {doc_id} {rank} {score} my_reranked_system\n")
                rank += 1
