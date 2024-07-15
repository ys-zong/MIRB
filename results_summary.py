import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resultJson', default='./results.json', type=str, help='Results json from evalutation.')
    return parser.parse_args()

args = parse_args()
with open(args.resultJson, "r") as f:
    data_dict = json.load(f)

# Define the groupings
groups = {
    "Knowledge": ["food", "sightseeing"],
    "Reasoning": ["codeu", "plot_code", "analogy", "3d_scene"],
    "Perception": ["image_jigsaw", "count", "attribute"],
    "Multi-Hop": ["visual_chain", "arxiv"]
}

def compute_averages(model_data, groups):
    averages = {}
    for group, features in groups.items():
        values = [model_data[feature] for feature in features]
        averages[group] = np.mean(values)
    return averages

# Compute the averages for each model
averages_dict = {model: compute_averages(data, groups) for model, data in data_dict.items()}

# Find the maximum values for each dimension
max_values = {dimension: max([averages[dimension] for averages in averages_dict.values()]) for dimension in groups.keys()}

for model, scores in averages_dict.items():
    formatted_scores = {cat: f"{score:.2f}" for cat, score in scores.items()}
    average_score = sum(scores.values()) / len(scores)
    print(f"{model}: {formatted_scores} (Average: {average_score:.2f})")
