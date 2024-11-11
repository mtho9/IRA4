from ranx import Qrels, Run, evaluate
import matplotlib.pyplot as plt

# Evaluate search results using ranx
# Version: November 5, 2024
# Authors: Abigail Pitcairn and Mandy Ho

# Function to print evaluation of results based on qrel
# Metrics: P@1, P@5, nDCG@5, MRR, MAP
def evaluate_search_result(qrel_file_path, result_file_path):
    # Load in files
    qrel = Qrels.from_file(qrel_file_path, kind="trec")
    run = Run.from_file(result_file_path, kind='trec')

    # Run tests and print results
    print(evaluate(qrel, run, "precision@1", make_comparable=True))
    print(evaluate(qrel, run, "precision@5", make_comparable=True))
    print(evaluate(qrel, run, "ndcg@5", make_comparable=True))
    print(evaluate(qrel, run, "mrr", make_comparable=True))
    print(evaluate(qrel, run, "map", make_comparable=True))


# Function to plot the ski jump graph
def plot_ski_jump(qrel_file_path, result_file_path, title="Ski Jump Plot", xlabel="Ranked Queries", ylabel="Precision"):

    # Generate data for ski jump plot
    qrel = Qrels.from_file(qrel_file_path, kind="trec")
    run = Run.from_file(result_file_path, kind='trec')
    data = (evaluate(qrel, run, "precision@5", return_mean=False, make_comparable=True))

    # Sort the data in descending order to create the ski jump effect
    sorted_data = sorted(data, reverse=True)

    # Plot the data
    plt.plot(sorted_data, marker='o', linestyle='-', color='b', label='Precision')

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show grid and legend
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()
