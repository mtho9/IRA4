import sys
import json
import re
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import numpy as np

class TFIDFModel:
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

    def clean_text(self, text):
        """Clean the input text by removing HTML, punctuation, and stop words."""
        text = self.remove_html(text)
        text = self.remove_punctuation(text).lower()
        text = self.remove_stop_words(text)
        return text

    def remove_html(self, text):
        """Remove HTML tags from the text."""
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text(separator=' ')

    def remove_punctuation(self, text):
        """Remove punctuation from the text."""
        return re.sub(r'[^\w\s]', '', text)

    def remove_stop_words(self, text):
        """Remove stop words from the text."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)

    def tf(self, lst: list) -> dict:
        """Calculate term frequency for each document."""
        tf_dic = defaultdict(lambda: defaultdict(int)) # Nested dictionary for term frequencies
        for doc in lst:
            docId = doc['Id']
            docText = doc['Text']
            terms = self.clean_text(docText).split()
            total_terms = len(terms)
            if total_terms == 0:
                continue # Skip if the document is empty
            for term in terms:
                tf_dic[term][docId] += 1
            for term in terms:
                tf_dic[term][docId] /= total_terms
        return dict(tf_dic) # Return the term frequency dictionary

    def idf(self, lst: list) -> dict:
        """Calculate inverse document frequency for each term."""
        idf_dic = {}
        total_docs = len(lst) # Total number of documents
        for item in lst:
            docId = item['Id']
            docText = item['Text']
            terms = self.clean_text(docText).split()
            unique_terms = set(terms)
            for term in unique_terms:
                # Increment the document count for each unique term
                if term in idf_dic:
                    idf_dic[term] += 1
                else:
                    idf_dic[term] = 1
        for term, doc_count in idf_dic.items():
            idf_dic[term] = np.log(total_docs / (1 + doc_count))
        return idf_dic

    def tf_idf(self, lst: list, query: dict, tf_dic: dict, idf_dic: dict) -> list:
        query_text = f"{query['Title']} {query['Body']}" # Combine query title and body
        query_terms = self.clean_text(query_text).split()
        result = {} # Dictionary to store scores for each document
        for doc in lst:
            docId = doc['Id']
            score = 0
            for qterm in query_terms:
                if qterm not in tf_dic or docId not in tf_dic[qterm]:
                    continue
                score += tf_dic[qterm][docId] * idf_dic[qterm]
            result[docId] = score
            # Sort results based on scores in descending order
        sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def search(self, query: dict, documents: list, tf_dic: dict, idf_dic: dict, top_k=100, tag="my_tfidf_system") -> None:
        """Search documents for a given query using the TF-IDF model and save results."""
        scores = self.tf_idf(documents, query, tf_dic, idf_dic)
        top_results = scores[:top_k]
        qid = query['Id']
        output_lines = []
        for rank, (doc_id, score) in enumerate(top_results, start=1):
            output_lines.append(f"{qid} Q0 {doc_id} {rank} {score:.6f} {tag}")
        with open("result_tfidf_2.tsv", "a") as file:
            for line in output_lines:
                file.write(line + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <documents_file.json> <queries_file.json>")
        sys.exit(1)

    documents_file = sys.argv[1]
    queries_file = sys.argv[2]

    with open(documents_file, 'r') as f:
        documents = json.load(f)

    with open(queries_file, 'r') as f:
        queries = json.load(f)

    # Initialize model
    tfidf_model = TFIDFModel()

    # Calculate TF and IDF for TF-IDF model
    tf_dic = tfidf_model.tf(documents)
    idf_dic = tfidf_model.idf(documents)

    total_queries = len(queries)

    for index, query in enumerate(queries):
        # Search with TF-IDF
        tfidf_model.search(query, documents, tf_dic, idf_dic)

        # Print progress
        percent_done = (index + 1) / total_queries * 100
        print(f"Processed {index + 1}/{total_queries} queries ({percent_done:.2f}%)")

    print("Search completed. Results saved in result_tfidf_1.tsv.")