import torch
from itertools import islice, cycle
import os
import random
import numpy as np
import json
import base64
from PIL import Image
import pandas as pd


def set_random_seed(seed_number):
    # position of setting seeds also matters
    os.environ['PYTHONHASHSEED'] = str(seed_number)
    np.random.seed(seed_number)
    random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.random.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def truncate_prediction(prediction: str) -> str:
    """Truncate captions at the first newline character, removing leading spaces."""
    prediction = prediction.strip()  # Remove leading and trailing whitespace
    trunc_index = prediction.find('\n')
    if trunc_index != -1:
        prediction = prediction[:trunc_index].strip()
    else:
        # If no newline is found, find the first period and truncate
        trunc_index = prediction.find('.') + 1
        if trunc_index > 0:
            prediction = prediction[:trunc_index].strip()
    return prediction


def load_image(img_ids, root_path):
    if isinstance(img_ids, str):
        img_ids = [img_ids]
    images = []
    image_paths = []
    for img_id in img_ids:
        image_path = os.path.join(root_path, img_id)
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        image_paths.append(image_path)
        
    return images, image_paths

## load data
def load_data(args, dataset):
    dataDir = args.dataDir
    query_file = os.path.join(dataDir, f'{dataset}.json')
    with open(query_file, 'r') as f:
        query_meta = json.load(f)

    return query_meta
    
def encode_image(image_path):
    _, file_extension = os.path.splitext(image_path)
    file_extension = file_extension.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }
    mime_type = mime_types.get(file_extension)
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image, mime_type

def get_task_instruction(args, dataset):
    if dataset in ['analogy', 'domain', 'plot', 'image_needles', 'plot_text', 'places', 'image_needles_concat']:
        instr = 'Answer with a single word.'
    elif dataset in ['codeu', 'foods', 'image_jigsaw', 'codeu_text']:
        instr = 'Answer with the option symbol.'
    elif dataset in ['arxiv', 'arxiv_text']:
        instr = 'Answer with the paper title.'
    elif dataset in ['count', 'count_concat']:
        instr = 'Answer with a single number.'
    elif dataset in ['3d_scene', '3d_scene_concat']:
        instr = 'The following images are different views of the same 3D scene. Answer with a single number.'
    
    return instr

def format_answer(answer, dataset, query=None):
    if dataset in ['count']:
        answer = str(answer)
    return answer