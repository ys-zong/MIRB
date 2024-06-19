import numpy as np
import re
import torch
import os

word_to_num = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", 
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10"
}


def eval_scores(results, dataset, model=None, tokenizer=None, processor=None):
    if dataset in ['count', 'codeu', 'foods', 'image_jigsaw', 'arxiv', 'image_needles', 'plot', '3d_scene', 
                   '3d_scene_concat', 'count_concat', 'image_needles_concat', 'plot_text', 'arxiv_text', 'codeu_text']:
        score = exact_match(results, dataset)
    elif dataset in ['analogy', 'domain']:
        score = exact_yes_no(results)
    elif dataset in ['places']:
        score = exact_in_match(results)
    return score

def exact_yes_no(results):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if result['answers'].lower() == 'yes' and 'yes' in str(prediction).lower():
            acc.append(1)
        elif result['answers'].lower() == 'no' and 'yes' not in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_in_match(results):
    acc = []
    for result in results:
        if result['answers'].lower() in ['yes', 'no']:
            prediction = result['prediction'].strip()
            prediction = prediction.strip('\n')
            trunc_index = prediction.find('\n')
            if trunc_index <= 0:
                trunc_index = prediction.find('.')
            if trunc_index > 0:
                prediction = prediction[:trunc_index]
            if result['answers'].lower() == 'yes' and 'yes' in str(prediction).lower():
                acc.append(1)
            elif result['answers'].lower() == 'no' and 'yes' not in str(prediction).lower():
                acc.append(1)
            else:
                acc.append(0)
            continue
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if str(result['answers']).lower() in str(prediction).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc

def exact_match(results, dataset):
    acc = []
    for result in results:
        prediction = result['prediction'].strip()
        prediction = prediction.strip('\n')
        trunc_index = prediction.find('\n')
        if trunc_index <= 0:
            trunc_index = prediction.find('.')
        if trunc_index > 0:
            prediction = prediction[:trunc_index]
        if dataset in ['count', 'image_needles', '3d_scene', '3d_scene_concat', 'count_concat', 'image_needles_concat']:
            # find the number
            match = re.search(r'\d+', prediction)
            if match:
                prediction = match.group()
            else:
                if str(prediction.lower()) in word_to_num:
                    prediction = word_to_num[str(prediction.lower())]
                else:
                    prediction = ''

        if str(prediction).lower() == str(result['answers']).lower():
            acc.append(1)
        else:
            acc.append(0)
    avg_acc = np.average(acc)
    return avg_acc
