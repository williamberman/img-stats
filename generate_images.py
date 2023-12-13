#!/usr/bin/env python

import os
import json
import logging
import torch
import uuid
from omegaconf import OmegaConf
import importlib
import numpy as np
import random
from tqdm import tqdm

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

config_file = os.environ.get('CONFIG_FILE', os.path.join(cur_dir, 'img_stats.yaml'))
config = OmegaConf.merge(
    dict(seed=0, batch_size=4, device='cuda'),
    OmegaConf.load(config_file), 
    OmegaConf.from_cli()
)
if config.save_to.type == 'local_fs':
    config.save_to.path = os.path.join(cur_dir, config.save_to.path, config.runner_id)
    os.makedirs(config.save_to.path, exist_ok=True)

def sub_config(path):
    config_ = config
    for x in path.split('.'):
        try:
            x = int(x)
        except:
            ...
        config_ = config_[x]
    return config_

model_config = sub_config(config.model_config)
stats_config = sub_config(config.stats_config)

def main():
    seed_all(config.seed)

    model = get_model()
    prompts = batch(get_prompts(), config.batch_size)
    prompts = [x for x in prompts]

    with Writer() as writer:
        for prompts_ in tqdm(prompts):
            images = model(prompts_, **model_config.args, **config.sweep_args).images
            writer.write(prompts_, images)

def seed_all(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available

def get_model():
    splitted = model_config.constructor.split('.')
    module = importlib.import_module('.'.join(splitted[:-2]))
    class_ = getattr(module, splitted[-2])
    constructor = getattr(class_, splitted[-1])
    model = constructor(**model_config.constructor_args).to(config.device)
    model.set_progress_bar_config(disable=True)
    return model

def get_prompts():
    assert stats_config.dataset == 'coco-validation-2017'
    prompt_file = os.path.join(cur_dir, "coco-validation-2017-prompts.jsonl")
    with open(prompt_file, 'r') as prompt_file:
        while True:
            line = prompt_file.readline()
            if line == '':
                break
            yield json.loads(line)

def batch(iter, batch_size):
    cur = []
    for i in iter:
        cur.append(i)
        if len(cur) == batch_size:
            yield cur
            cur = []
    if len(cur) > 0:
        yield cur

class Writer:
    def __init__(self):
        assert config.save_to.type == 'local_fs'

    def __enter__(self):
        self.index_file = open(os.path.join(config.save_to.path, 'index.jsonl'), 'w')
        return self

    def write(self, prompts, images):
        for prompt, image in zip(prompts, images):
            filename = str(uuid.uuid4())
            filepath = os.path.join(config.save_to.path, f"{filename}.png")
            image.save(filepath)
            self.index_file.write(json.dumps([filename, prompt]) + '\n')

    def __exit__(self, exc_type, exc_value, traceback):
        self.index_file.close()

if __name__ == "__main__":
    main()