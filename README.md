### Author
Mandy Ho

###Overview
This project implements an Information Retrieval system that retrieves and reranks the top 100 search results for each query from a collection of documents.

###Steps
BM25 Search: The initial step involves using a BM25 search to retrieve the top 100 results for each query from a document collection. This was completed in Assignment 2.
Reranking with LLM: In the second step, the results from the BM25 search are reranked using a LLM. The reranking is done using pointwise query generation prompts to improve result relevance.

###Requirements
Python 3.x

###Required Libraries
torch
transformers
re
json
bs4 (BeautifulSoup)
os
sys
torch.nn.functional
LLMRerankSearch

###How to Run
python Main.py Answers.json topics_1.json topics_2.json

###Additional Files
bm25_1.tsv: The BM25 search results from the first set of queries.
bm25_2.tsv: The BM25 search results from the second set of queries.
