import torch
import os
import json
import argparse
import gc
import numpy as np
from utils import model_inference, utils, load_models


mcq_tasks = ['codeu', 'food', 'image_jigsaw', 'codeu_text']
generation_tasks = ['analogy', 'count', 'arxiv', 'attribute', 'visual_chain', 'plot_code', 'sightseeing', "3d_scene", '3d_scene_concat', 'image_needles_concat', 'count_concat',
                    'plot_text', 'arxiv_text']

def parse_args():
    parser = argparse.ArgumentParser(description='I2T Inference')

    parser.add_argument('--dataDir', default='./MIR', type=str, help='Data directory.')
    parser.add_argument('--dataset', default=['count'], choices=['analogy', 'count', 'arxiv', 'attribute', 'visual_chain', 'plot_code', 'codeu', 'sightseeing', 'food', 'image_jigsaw', "3d_scene"
                                                                 , '3d_scene_concat', 'visual_chain_concat', 'count_concat', 'plot_text', 'arxiv_text', 'codeu_text'], nargs="+")
    parser.add_argument("--engine", "-e", choices=["llava15-7b", "llava16-7b", "llava16-13b", "qwen-vl", "qwen-vl-chat", 'internlm-x2', 'longva-7b',
                                                   'emu2-chat', 'idefics1-9b-instruct', 'idefics2-8b', 'mantis-idefics2', 'gpt4v', 'vila-7b', 'vila-2.7b',
                                                   "phi3-vision", "gemini-pro", "internlm-x2d5"],
                        default=["phi3-vision"], nargs="+")
    
    parser.add_argument('--max-new-tokens', default=15, type=int, help='Max new tokens for generation.')
    parser.add_argument('--CoT', default=False, action='store_true', help='Whether to use CoT.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    return parser.parse_args()


def eval_CoT(args, query_meta, model, tokenizer, processor, engine, dataset):
    data_path = args.dataDir

    for query in query_meta:
        query_text = query['questions']
        query['questions'] = query['questions'] + "Do not answer the question. Give concise reasoning steps of solving this problem."
        max_new_tokens = 250
        predicted_CoT = model_inference.ICL_I2T_inference(args, engine, dataset, model, tokenizer, query, 
                                                       data_path, processor, max_new_tokens)
        query['questions'] = query_text + "\nHere's the potential logics of answering this question: " + predicted_CoT + "\nNow, directly answer the question."

    return query_meta

def eval_gen(args, query_meta, model, tokenizer, processor, engine, dataset):
    data_path = args.dataDir
    results = []
    max_new_tokens = args.max_new_tokens

    for query in query_meta:
        predicted_answer = model_inference.ICL_I2T_inference(args, engine, dataset, model, tokenizer, query, 
                                                       data_path, processor, max_new_tokens)
        query['prediction'] = predicted_answer
        results.append(query)

    return results

def eval_mcq(args, query_meta, model, tokenizer, processor, engine, dataset):
    data_path = args.dataDir
    results = []
    choices = ['A', 'B', 'C', 'D', 'E']
    for query in query_meta:
        token_probs = model_inference.I2T_first_prob(args, engine, dataset, model, tokenizer, query, 
                                                      data_path, processor)
        lprobs = []
        for ans in choices:
            ans_id = tokenizer(ans, add_special_tokens=False, return_tensors="pt").input_ids[0].item()
            lprobs.append(token_probs[ans_id].item())
        
        predicted_answer = {0: "A", 1: "B", 2: "C", 3: "D", 4: 'E'}[np.argmax(lprobs)]
        query['prediction'] = predicted_answer
        results.append(query)

    return results
    

if __name__ == "__main__":
    args = parse_args()
    
    for engine in args.engine:
        
        model, tokenizer, processor = load_models.load_i2t_model(engine, args)
        print("Loaded model: {}\n".format(engine))
        
        utils.set_random_seed(args.seed)
        for dataset in args.dataset:
            query_meta = utils.load_data(args, dataset)
            if args.CoT:
                query_meta = eval_CoT(args, query_meta, model, tokenizer, processor, engine, dataset)
            if dataset in mcq_tasks and engine not in ['gpt4v', 'emu2-chat']:
                results_dict = eval_mcq(args, query_meta, model, tokenizer, processor, engine, dataset)
            elif dataset in generation_tasks or engine in ['gpt4v', 'emu2-chat']:
                results_dict = eval_gen(args, query_meta, model, tokenizer, processor, engine, dataset)

            os.makedirs(f"results/{dataset}", exist_ok=True)
            if args.CoT:
                result_path = f"results/{dataset}/{engine}_CoT.json"
            else:
                result_path = f"results/{dataset}/{engine}.json"
            with open(result_path, "w") as f:
                json.dump(results_dict, f, indent=4)

        del model, tokenizer, processor
        torch.cuda.empty_cache()
        gc.collect()