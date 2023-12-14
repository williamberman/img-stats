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
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
import wandb

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))
OmegaConf.register_new_resolver('wurst_stage_c_timesteps', lambda : DEFAULT_STAGE_C_TIMESTEPS)

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

config_file = os.environ.get('CONFIG_FILE', os.path.join(cur_dir, 'img_stats.yaml'))
config = OmegaConf.merge(
    dict(),
    OmegaConf.load(config_file), 
    OmegaConf.from_cli(),
)

def main():
    for model_idx in range(len(config.models)):
        for sweep_args in get_sweep_args(config.models[model_idx].get('sweep_args', {})):
            for stats_idx in range(len(config.stats)):
                model_config = f"models.{model_idx}"
                stats_config = f"stats.{stats_idx}"

                run_suffix = get_run_suffix(model_config, sweep_args)
                runner_id = f"{config.run_prefix}_{run_suffix}"

                assert config.save_to.type == 'local_fs'
                run_save_path = os.path.join(cur_dir, config.save_to.path, runner_id)
                filename = os.path.join(run_save_path, 'metrics.json')

                try:
                    with open(filename) as f:
                        print(json.load(f))
                except FileNotFoundError:
                    print(f"file not found: {filename}")

def get_sweep_args(sweep_args):
    res = [{}]
    keys = sweep_args.keys()
    for k in keys:
        new = []
        for v in sweep_args[k]:
            for cur in res:
                new.append({**cur, k: v})
        res = new
    return res

def get_run_suffix(model_config, sweep_args):
    run_suffix = model_config

    sweep_arg_keys = [x for x in sweep_args.keys()]
    sweep_arg_keys.sort()

    for k in sweep_arg_keys:
        run_suffix += f"_{k}_{sweep_args[k]}"

    return run_suffix

if __name__ == "__main__":
    main()