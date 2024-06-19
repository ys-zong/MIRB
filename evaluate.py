import torch
import json
import argparse
from evals import eval
import os


metrics = {
    "analogy": "Acc",
    "codeu": "Acc",
    "count": "Acc",
    "arxiv": "Acc",
    "domain": "Acc",
    "image_needles": "Acc",
    "plot": "Acc",
    "places": "Acc",
    "foods": "Acc",
    "image_jigsaw": "Acc",
    "3d_scene": "Acc",
    "codeu_text": "Acc",
    "count_concat": "Acc",
    "plot_text": "Acc",
    "arxiv_text": "Acc",
    "3d_scene_concat": "Acc",
    "image_needles_concat": "Acc",
}


def parse_args():
    parser = argparse.ArgumentParser(description='I2T Evaluation')

    parser.add_argument('--dataset', default='analogy', choices=['analogy', 'codeu', 'count', 'arxiv', 'domain', 'image_needles', 'plot', 'places', 'foods', 'image_jigsaw', '3d_scene',
                                                                 'codeu_text', 'count_concat', 'plot_text', 'arxiv_text', '3d_scene_concat', 'image_needles_concat'], nargs="+")
    parser.add_argument("--engine", "-e", choices=["llava16-7b", "llava16-13b", "llava15-7b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 
                                                   'emu2-chat', 'idefics1-9b-instruct', 'idefics1-80b-instruct', 'idefics2-8b', 'gpt4v', 'vila-7b', 'vila-2.7b',
                                                   "phi3-vision"],
                        default=["phi3-vision"], nargs="+")
    parser.add_argument("--resultJson", "-r", default="results.json", type=str, help="Result json file.")
    parser.add_argument('--CoT', default=False, action='store_true', help='Whether to eval CoT.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.resultJson = args.resultJson.split('.json')[0] + '_CoT' + '.json' if args.CoT else args.resultJson
    if os.path.exists(args.resultJson):
        with open(args.resultJson, "r") as f:
            results_all = json.load(f)
    else:
        results_all = {}

    for engine in args.engine:
        if engine not in results_all:
            results_all[engine] = {}
        for dataset in args.dataset:
            if args.CoT:
                result_file = f"results/{dataset}/{engine}_CoT.json"
            else:
                result_file = f"results/{dataset}/{engine}.json"
            with open(result_file, "r") as f:
                results_dict = json.load(f)

            score = eval.eval_scores(results_dict, dataset)
            score = round(score * 100.0, 2) # keep 2 decimal places
            print(f'{dataset} {metrics[dataset]} of {engine}: ', f"{score}", flush=True)
            results_all[engine][dataset] = score

    with open(args.resultJson, "w") as f:
        json.dump(results_all, f, indent=4)
