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
import PIL.Image
from transformers import CLIPModel, CLIPProcessor
import math
import torch_fidelity

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

config_file = os.environ.get('CONFIG_FILE', os.path.join(cur_dir, 'img_stats.yaml'))
config = OmegaConf.merge(
    dict(seed=0, generate_images_batch_size=4, clip_score_batch_size=16, device='cuda', generate_images=True, take_metrics=True, sweep_args={}),
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

model_config = OmegaConf.merge(
    dict(args={}),
    sub_config(config.model_config)
)

stats_config = OmegaConf.merge(
    dict(total_images=math.inf),
    sub_config(config.stats_config)
)

def main():
    seed_all(config.seed)

    logger.warning(f"config.device: {config.device}")

    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logger.warning(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logger.warning('CUDA_VISIBLE_DEVICES not set')

    if config.generate_images:
        logger.warning("generating images")

        model = get_model()
        prompts = tqdm([x for x in batch(get_prompts(), config.generate_images_batch_size)])

        with ImagesWriter() as writer:
            for prompts_ in prompts:
                images = model(prompts_, **model_config.args, **config.sweep_args).images
                for prompt, image in zip(prompts_, images):
                    writer.write(prompt, image)

    if config.take_metrics:
        logger.warning("taking metrics")

        metrics = {}

        if 'clip' in stats_config.metrics:
            metrics['clip'] = compute_clip()

        do_isc = 'isc' in stats_config.metrics
        do_fid = 'fid' in stats_config.metrics

        if do_isc or do_fid:
            assert stats_config.dataset == 'coco-validation-2017'
            assert config.save_to.type == 'local_fs'

            for filename in os.listdir(config.save_to.path):
                if filename.endswith('.png'):
                    samples_resize_and_crop = PIL.Image.open(os.path.join(config.save_to.path, filename)).height
                    break

            # TODO - we should be able to just calculate the metrics for coco_val_2017
            # once and then re-compute fid/etc.. against those.
            metrics.update(
                **torch_fidelity.calculate_metrics(
                    input1=config.save_to.path, 
                    input2=os.path.join(cur_dir, 'coco_val_2017'),
                    cuda=True, 
                    isc=do_isc, 
                    fid=do_fid, 
                    samples_resize_and_crop=samples_resize_and_crop,
                    verbose=True,
                )
            )

        assert config.save_to.type == 'local_fs'
        with open(os.path.join(config.save_to.path, 'metrics.json'), 'w') as metrics_file:
            json.dump(metrics, metrics_file)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # it is safe to call this function even if cuda is not available

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
    ctr = 0
    with open(prompt_file, 'r') as prompt_file:
        while True:
            line = prompt_file.readline()
            if line == '' or ctr > stats_config.total_images:
                break
            yield json.loads(line)
            ctr += 1

def batch(iter, batch_size):
    cur = []
    for i in iter:
        cur.append(i)
        if len(cur) == batch_size:
            yield cur
            cur = []
    if len(cur) > 0:
        yield cur

class ImagesWriter:
    def __init__(self):
        assert config.save_to.type == 'local_fs'

    def __enter__(self):
        # buffering=1 -> line buffered
        filename = os.path.join(config.save_to.path, 'index.jsonl')
        logger.warning(f"writing to index file {filename}")
        self.index_file = open(filename, 'w', buffering=1)
        return self

    def write(self, prompt, image):
        filename = str(uuid.uuid4()) + ".png"
        filepath = os.path.join(config.save_to.path, filename)
        image.save(filepath)
        self.index_file.write(json.dumps([filename, prompt]) + '\n')

    def __exit__(self, exc_type, exc_value, traceback):
        self.index_file.close()


def get_images():
    assert config.save_to.type == 'local_fs'
    with open(os.path.join(config.save_to.path, 'index.jsonl'), 'r') as index_file:
        while True:
            line = index_file.readline()
            if line == '':
                break
            filename, prompt = json.loads(line)
            filepath = os.path.join(config.save_to.path, filename)
            yield filepath, prompt

def compute_clip():
    logger.warning('computing clip')

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    clip.to(config.device)

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    images_ = tqdm([x for x in batch(get_images(), config.clip_score_batch_size)])

    clip_scores_sum = 0
    num_images = 0

    for x in images_:
        prompts = []
        images = []
        for filename, prompt in x:
            images.append(PIL.Image.open(filename))
            prompts.append(prompt)
        input = clip_processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        input["pixel_values"] = input["pixel_values"].to(dtype=torch.float16, device=config.device)
        input["input_ids"] = input["input_ids"].to(config.device)
        input["attention_mask"] = input["attention_mask"].to(config.device)

        clip_scores = clip(**input).logits_per_image.diag()
        clip_scores_sum += clip_scores.sum().item()
        num_images += len(clip_scores)

        images_.set_postfix(clip_score=clip_scores_sum/num_images)

    images_.close()

    return clip_scores_sum/num_images

if __name__ == "__main__":
    main()