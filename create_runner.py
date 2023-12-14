#!/usr/bin/env python

import os
import logging
import torch
from omegaconf import OmegaConf
import stat
from random import choice, randint

# https://github.com/ferhatelmas/pyhaikunator/blob/e55c66393ed933d598489150ec2c20c70d190a72/haikunator.py

adjectives = """
    autumn hidden bitter misty silent empty dry dark summer
    icy delicate quiet white cool spring winter patient
    twilight dawn crimson wispy weathered blue billowing
    broken cold damp falling frosty green long late lingering
    bold little morning muddy old red rough still small
    sparkling throbbing shy wandering withered wild black
    young holy solitary fragrant aged snowy proud floral
    restless divine polished ancient purple lively nameless
""".split()

nouns = """
    waterfall river breeze moon rain wind sea morning
    snow lake sunset pine shadow leaf dawn glitter forest
    hill cloud meadow sun glade bird brook butterfly
    bush dew dust field fire flower firefly feather grass
    haze mountain night pond darkness snowflake silence
    sound sky shape surf thunder violet water wildflower
    wave water resonance sun wood dream cherry tree fog
    frost voice paper frog smoke star
""".split()

def gen_id(token_range=9999, delimiter='-'):
    if not isinstance(token_range, int) or token_range < 0:
        raise RuntimeError('Token range must be a nonnegative integer')
    if not isinstance(delimiter, str):
        raise RuntimeError('Delimiter must be a string')
    res = [choice(adjectives), choice(nouns)]
    r = randint(0, token_range)
    if r != 0:
        res.append(str(r))
    return delimiter.join(res)

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

config_file = os.environ.get('CONFIG_FILE', os.path.join(cur_dir, 'img_stats.yaml'))
config = OmegaConf.merge(
    dict(gpus=8, write_to="runner.sh", run_in_background=True, slurm=False, run_prefix=gen_id(delimiter='_')),
    OmegaConf.load(config_file), 
    OmegaConf.from_cli()
)
config.log_dir = os.path.join(cur_dir, config.log_dir)

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

def get_runner_id(model_config, sweep_args):
    runner_id = f"{config.run_prefix}_{model_config}"

    sweep_arg_keys = [x for x in sweep_args.keys()]
    sweep_arg_keys.sort()

    for k in sweep_arg_keys:
        runner_id += f"_{k}_{sweep_args[k]}"

    return runner_id

def main():
    os.makedirs(config.log_dir, exist_ok=True)

    with open(config.write_to, 'w') as f:
        gpu = 0

        f.write('#! /bin/bash \n\n')

        for model_idx in range(len(config.models)):
            for sweep_args in get_sweep_args(config.models[model_idx].get('sweep_args', {})):
                for stats_idx in range(len(config.stats)):
                    model_config = f"models.{model_idx}"
                    stats_config = f"stats.{stats_idx}"

                    runner_id = get_runner_id(model_config, sweep_args)
                    logfile = os.path.join(config.log_dir, runner_id + '.log')

                    args = f'model_config={model_config} stats_config={stats_config} runner_id={runner_id}'
                    cmd = f"/usr/bin/time -f '%E real,%U user,%S sys' {os.path.join(cur_dir, 'img_stats.py')} {args}"

                    if config.slurm:
                        cmd += f' sweep_args=\\"{sweep_args}\\"'
                        sbatch_cmd = "sbatch --ntasks=1 --cpus-per-task=12 --gpus-per-task=1"
                        full_cmd = f"{sbatch_cmd} --output={logfile} --wrap \"{cmd}\""
                    else:
                        cmd += f' sweep_args="{sweep_args}"'

                        envvars = f"CUDA_VISIBLE_DEVICES={gpu}"

                        full_cmd = f"{envvars} {cmd}"

                        if config.run_in_background:
                            full_cmd += f" &> {logfile} &"

                    f.write(full_cmd + '\n\n')

                    gpu = (gpu + 1) % config.gpus

    st = os.stat(config.write_to)
    os.chmod(config.write_to, st.st_mode | stat.S_IEXEC)

if __name__ == "__main__":
    main()