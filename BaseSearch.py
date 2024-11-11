import json
import math
import re
from bs4 import BeautifulSoup
from collections import defaultdict, Counter

# Implement TF-IDF search from assignment 2 for base results
# Version: November 12, 2024
# Authors: Abigail Pitcairn and Mandy Ho


# TF-IDF search from a documents file path and queries file path
def base_search(queries_file_path, documents_file_path):
    print("Conducting tf-idf search...")
    docs = load_json_file(documents_file_path)
    inverted_index = build_inverted_indexes(docs)
    return query_load_and_search(queries_file_path, inverted_index)


# Specify how many documents you would like returned in the result file
total_return_documents = 100


# Define stop words to be removed from index
stop_words = ["the", "of", "and", "to", "a", "in", "is", "that", "was", "it", "for", "on", "with", "he", "be",
              "I", "by", "as", "at", "you", "are", "his", "had", "not", "this", "have", "from", "but", "which", "she",
              "they", "or", "an", "her", "were", "there", "we", "their", "been", "has", "will", "one", "all",
              "would", "can", "if", "who", "more", "when", "said", "do", "what", "about", "its", "it's", "so", "up",
              "into", "no", "him", "some", "could", "them", "only", "time", "out", "my", "two", "other", "then", "may",
              "over", "also", "new", "like", "these", "me", "after", "first", "your", "did", "now", "any", "people",
              "than", "should", "very", "most", "see", "where", "just", "made", "between", "back", "way", "many",
              "years", "being", "our", "how", "work"]


# Load the JSON file
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Clean and tokenize the text and remove stop words
def clean_and_tokenize(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    tokens = re.findall(r'\b\w+\b', clean_text.lower())
    filtered_words = [word for word in tokens if word.lower() not in stop_words]
    return filtered_words


# Build the inverted index with tf-idf values
def build_inverted_indexes(docs):
    print("Building inverted index for base search...")
    inverted_index_tfidf = defaultdict(lambda: defaultdict(float))
    df = dfs(docs)
    for doc in docs:
        if 'Text' not in doc:
            print(f"Missing 'Text' in document: {doc}")
        doc_id = doc['Id']
        tokens = clean_and_tokenize(doc['Text'])
        for token in set(tokens):
            inverted_index_tfidf[token][doc_id] = tf_idf(token, tokens, df, docs)
    return inverted_index_tfidf


# Calculate term frequency for a term-document pair
def tf(term, doc_tokens):
    term_counts = Counter(doc_tokens)
    return term_counts[term]


# Calculate document frequencies and store them in a dictionary for quick use
def dfs(docs):
    df = defaultdict(int)
    for doc in docs:
        if 'Text' not in doc:
            print(f"Missing 'Text' in document: {doc}")
        unique_terms = set(clean_and_tokenize(doc['Text']))
        for term in unique_terms:
            df[term] += 1
    return df


# Calculate the inverse document frequency for a term and collection
def idf(term, df, docs):
    n = len(docs) + 1
    return math.log(n / (df[term] + 1))


# Calculate term frequency-inverse document frequency
def tf_idf(term, doc_tokens, df, docs):
    return tf(term, doc_tokens) * idf(term, df, docs)


# Return the set of document IDs ranked by score for the query
def search(q, inverted_index):
    result = {}
    terms = clean_and_tokenize(q)
    for term in terms:
        if term in inverted_index:
            for doc_id, score in inverted_index[term].items():
                result[doc_id] = result.get(doc_id, 0.0) + score
    return {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}


# Write the input results to an output file
def save_to_result_file(results, output_file):
    with open(output_file, 'w') as f:
        for query_id in results:
            dic_result = results[query_id]
            rank = 1
            for doc_id in dic_result:
                f.write(f"{query_id} 0 {doc_id} {rank} {dic_result[doc_id]} Run1\n")
                rank += 1
                if rank > total_return_documents:
                    break


# Load the queries from the topics file and perform search
def query_load_and_search(topics, inverted_index):
    print("Conducting base search...")
    queries = load_json_file(topics)
    search_results = {}
    for query_data in queries:
        query_id = query_data['Id']
        title = query_data['Title']
        body = query_data['Body']
        query_text = title + " " + body
        result_ids = search(query_text, inverted_index)
        search_results[query_id] = result_ids
    return search_results
