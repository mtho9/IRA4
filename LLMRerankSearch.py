import re
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from torch import autocast

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


def generate_prompt_for_qg(query_id, doc_ids, topics, answers):
    """Generate a prompt for the LLM to generate the query from the document."""

    # Define the system message to guide the LLM
    system_message = "You are a relevance document judger for a search engine. " \
                     "For each document, generate the query based on the document's content. " \
                     "Then, calculate the relevance of the document to the original query based on how closely the generated query matches the original query."

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
        prompt = f"Query: {query_text}\nDocument: {doc_text}\nGenerate a query based on the document's content and assess the relevance of this document to the query."
        prompts.append(prompt)

    return prompts


def process_batch_for_qg(doc_batch, pipeline, query_text):
    """Process a batch of documents using the LLM pipeline and compute query likelihood."""
    batch_scores = {}

    # Reduce the batch size to avoid OOM
    batch_size = 2  # Decrease the batch size to reduce memory usage
    for i in range(0, len(doc_batch), batch_size):
        batch = doc_batch[i:i + batch_size]  # Create smaller batches

        with autocast():  # Enable mixed precision
            outputs = pipeline(batch, max_new_tokens=50, temperature=0.6, top_p=0.9,
                               pad_token_id=pipeline.tokenizer.eos_token_id)

        for j, output in enumerate(outputs):
            generated_text = output[0]['generated_text']

            # Tokenize both the original query and generated query once to avoid redundant tokenization
            input_ids = pipeline.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(
                pipeline.device)
            generated_ids = pipeline.tokenizer(generated_text, return_tensors="pt", padding=True,
                                               truncation=True).input_ids.to(pipeline.device)

            with torch.no_grad():
                # Compute the log-likelihood of the generated query against the original query
                model_outputs = pipeline.model(input_ids=input_ids, labels=generated_ids)
                log_likelihood = model_outputs.loss.item()  # This is the negative log-likelihood loss

            # Assign the log-likelihood as the relevance score
            batch_scores[j] = -log_likelihood  # Negative because lower loss means more relevant

        # Clear GPU memory after each batch to prevent OOM
        torch.cuda.empty_cache()

    return batch_scores


def rerank_documents_with_qg(doc_results, topics, answers, pipeline, tokenizer, device):
    reranked_docs = {}

    # Ensure the pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Ensure pad_token is set

    # Loop through each topic and its corresponding documents
    for topic_id, doc_ids in doc_results.items():
        # Get the query for the current topic
        query = next((topic for topic in topics if topic['Id'] == topic_id), None)
        if not query:
            continue

        query_text = f"{query['Title']} {query['Body']}"

        # Prepare a list to hold the document texts
        document_texts = []
        for doc_id in doc_ids:
            doc_text = next((answer['Text'] for answer in answers if answer['Id'] == doc_id), "")
            document_texts.append(f"Query: {query_text} \nDocument: {doc_text}")

        # Generate queries for each document using the LLM
        generated_queries = pipeline(document_texts, max_new_tokens=50, temperature=0.6, top_p=0.9)

        # Now calculate the relevance score for each document
        similarities = []
        for i, generated_query in enumerate(generated_queries):
            generated_query_text = generated_query[0]['generated_text']  # Get the generated text

            # Calculate cosine similarity between the original query and the generated query
            query_emb = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True).to(device)
            generated_query_emb = tokenizer(generated_query_text, return_tensors="pt", padding=True,
                                             truncation=True).to(device)

            with torch.no_grad():
                # Get the logits for the query and generated query
                query_output = pipeline.model(**query_emb)
                generated_query_output = pipeline.model(**generated_query_emb)

                # Get logits from the model (usually this is what you would use in causal language models)
                query_logits = query_output.logits.mean(dim=1)  # Averaging logits over tokens
                generated_query_logits = generated_query_output.logits.mean(dim=1)  # Averaging logits over tokens

                # Calculate cosine similarity between the original query logits and the generated query logits
                cosine_sim = F.cosine_similarity(query_logits, generated_query_logits)

            # Append the score along with document ID
            similarities.append((doc_ids[i], cosine_sim.item()))

        # Sort documents by the similarity score
        reranked_docs[topic_id] = sorted(similarities, key=lambda x: x[1], reverse=True)

        # Clear variables that are no longer needed to free up memory
        del document_texts
        del generated_queries
        del similarities

        # Clear GPU memory after processing each topic
        torch.cuda.empty_cache()

    return reranked_docs

def write_ranked_results(query_results, output_file):
    with open(output_file, 'w') as file:
        for query_id, documents in query_results.items():
            rank = 1
            for doc_id, score in documents:
                # Write in the required format
                file.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} my_bm25_system\n")
                rank += 1