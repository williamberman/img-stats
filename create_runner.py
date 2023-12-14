#!/usr/bin/env python

import os
import logging
import torch
from omegaconf import OmegaConf
import stat
from random import choice, randint
import math

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

cli_config = OmegaConf.merge(dict(config_file="img_stats.yaml"), OmegaConf.from_cli())
cli_config.config_file = os.path.join(cur_dir, cli_config.config_file)

config = OmegaConf.merge(
    dict(gpus=8, run_in_background=True, slurm=False, run_prefix=gen_id(delimiter='_')),
    OmegaConf.load(cli_config.config_file), 
   cli_config 
)
config.log_dir = os.path.join(cur_dir, config.log_dir)

if 'write_to' not in config:
    config.write_to = f'runner_{config.run_prefix}.slurm' if config.slurm else f'runner_{config.run_prefix}.sh'

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


def main():
    os.makedirs(config.log_dir, exist_ok=True)

    cmds = make_cmds()

    logger.warning(f"writing {len(cmds)} commands to: {config.write_to}")

    with open(config.write_to, 'w') as f:
        if config.slurm:
            f.write('#! /bin/bash \n')
            f.write(f'#SBATCH --nodes {int(math.ceil(float(len(cmds)) / 8.0))}\n')
            f.write(f'#SBATCH --ntasks {len(cmds)}\n')
            f.write('#SBATCH --cpus-per-task=12\n')
            f.write('#SBATCH --gpus-per-task=1\n')
            f.write('#SBATCH --exclusive\n')
            f.write('#SBATCH --partition=production-cluster\n')
            f.write('\n')
            f.write('set -e -u \n\n')
        else:
            f.write('#! /bin/bash \n\n')
            f.write('set -e -u \n\n')

        f.write('echo "starting runner script"\n\n')

        for cmd in cmds:
            f.write(f"{cmd}\n\n")

        if config.slurm:
            f.write('echo "finished queueing jobs. waiting for completion" \n')
            f.write('wait\n')
            f.write('echo "all jobs finished. exiting."\n')
        else:
            if config.run_in_background:
                f.write('echo "all jobs running in background, exiting runner script but jobs are still running."\n')
            else:
                f.write('echo "all jobs done."\n')

    if not config.slurm:
        st = os.stat(config.write_to)
        os.chmod(config.write_to, st.st_mode | stat.S_IEXEC)

    logger.warning(f"output written to: {config.write_to}")

def make_cmds():
    gpu = 0

    cmds = []

    for model_idx in range(len(config.models)):
        for sweep_args in get_sweep_args(config.models[model_idx].get('sweep_args', {})):
            for stats_idx in range(len(config.stats)):
                model_config = f"models.{model_idx}"
                stats_config = f"stats.{stats_idx}"

                run_suffix = get_run_suffix(model_config, sweep_args)
                runner_id = f"{config.run_prefix}_{run_suffix}"
                logfile = os.path.join(config.log_dir, runner_id + '.log')

                args = f'config_file={config.config_file} model_config={model_config} stats_config={stats_config} run_prefix={config.run_prefix} run_suffix={run_suffix} runner_id={runner_id} sweep_args="{sweep_args}"'

                for k, v in config.models[model_idx].get('img_stats_args', {}).items():
                    args += f' {k}={v}'

                cmd = f"/usr/bin/time -f '%E real,%U user,%S sys' {os.path.join(cur_dir, 'img_stats.py')} {args}"

                if config.slurm:
                    full_cmd = f"srun --ntasks=1 --nodes=1 --exclusive --output={logfile} {cmd} &"
                else:
                    full_cmd = f"CUDA_VISIBLE_DEVICES={gpu} {cmd}"

                    if config.run_in_background:
                        full_cmd += f" &> {logfile} &"

                cmds.append(full_cmd)

                gpu = (gpu + 1) % config.gpus

    return cmds

if __name__ == "__main__":
    main()