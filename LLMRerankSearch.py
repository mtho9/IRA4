import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

stop_words = set([
    # List of stop words
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they',
    'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is',
    'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
    'd', 'll', 'm', 'o', 're',
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
    'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
    "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", '-', '.'
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
    """Read the results from a file."""
    topicdict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            topic_id = parts[0]
            doc_id = parts[2]

            if topic_id not in topicdict:
                topicdict[topic_id] = []
            if len(topicdict[topic_id]) < 100:  # limit to 100 documents per topic
                topicdict[topic_id].append(doc_id)

    return topicdict


def generate_prompt(query_id, doc_ids, topics, answers):
    """Generate a prompt for the LLM to rank document relevance based on the query."""

    # Define the system message to guide the LLM
    system_message = "You are a relevance document judger for a search engine. " \
                     "For each document, rank its relevance to the query from 1 (least relevant) to 5 (most relevant). " \
                     "You will only output the relevance score."

    # Retrieve the query details from the topics
    query = next((topic for topic in topics if topic['Id'] == query_id), None)
    if not query:
        return []

    topic_title = query['Title']
    topic_body = query['Body']
    query_text = f"{topic_title} {topic_body}"

    # Get the relevant documents
    prompts = []
    for doc_id in doc_ids:
        doc_text = next((answer['Text'] for answer in answers if answer['Id'] == doc_id), "")
        prompt = f"Query: {query_text}\nDocument: {doc_text}\nRank the relevance of this document from 1 (least relevant) to 5 (most relevant)."
        prompts.append(prompt)

    return prompts


def process_batch(doc_batch, pipeline):
    """Process a batch of documents using the LLM pipeline."""
    outputs = pipeline(doc_batch, max_new_tokens=50, temperature=0.6, top_p=0.9,
                       pad_token_id=pipeline.tokenizer.eos_token_id)

    batch_scores = {}
    for i, output in enumerate(outputs):
        generated_text = output[0]['generated_text']
        try:
            model_score = int(generated_text.strip())
        except ValueError:
            model_score = 0  # Default to 0 if parsing fails
        batch_scores[i] = model_score

    return batch_scores


def rerank_documents_with_llm(topicdict, topics, answers, pipeline, batch_size=8) -> dict:
    """Rerank the documents using the LLM with parallelized processing."""
    reranked_results = {}

    with ThreadPoolExecutor() as executor:
        for topic_id, doc_ids in topicdict.items():
            prompts = generate_prompt(topic_id, doc_ids, topics, answers)
            doc_batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

            # Run the batches concurrently
            futures = [executor.submit(process_batch, batch, pipeline) for batch in doc_batches]

            scores = {}
            for future in futures:
                batch_scores = future.result()
                for idx, score in batch_scores.items():
                    doc_id = doc_ids[idx]
                    scores[doc_id] = score

            reranked_results[topic_id] = {doc_id: score for doc_id, score in
                                          sorted(scores.items(), key=lambda x: x[1], reverse=True)}

    return reranked_results


def write_ranked_results(query_results, output_file):
    """Write ranked results to a file."""
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, score in documents.items():
                file.write(f"{query_id} Q0 {doc_id} {rank} {score} my_reranked_system\n")
                rank += 1