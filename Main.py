import sys
import time
import BaseSearch
import Evaluation
import LLM
import LLMSearch

# Main method to run COS470 Intro to IR Assignment 4
# Retrieve documents from collection for a set of queries
# Search with base model then rerank with LLM
# Version: November 12, 2024
# Authors: Mandy Ho and Abigail Pitcairn

def main(answers, topics_1, topics_2):
    print("Starting main...")

    start = time.time()

    base_results = BaseSearch.base_search(topics_1, answers)

    BaseSearch.save_to_result_file(base_results, "base_result.tsv")
    end = time.time()
    Evaluation.evaluate_search_result("qrel_1.tsv","base_result.tsv")
    print(f"Base search time: {end-start}")

    # Rerank with LLM

    # Evaluate results


# Terminal Command: python3 Main.py Answers.json topics_1.json
# OR python3 Main.py Answers.json topics_2.json
if __name__ == "__main__":
    # Manual run command because I'm lazy
    main("Answers.json", "topics_1.json", "topics_2.json")

    # Uncomment this part for submission
    # if len(sys.argv) != 4:
    #     print("Usage: python main.py <answers.json> <topics_1.json> <topics_2.json>")
    #     sys.exit(1)
    #
    # answers_file = sys.argv[1]
    # topics_1_file = sys.argv[2]
    # topics_2_file = sys.argv[3]
    #
    # main(answers_file, topics_1_file, topics_2_file)