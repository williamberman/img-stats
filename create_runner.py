import os
import logging
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

config_file = os.environ.get('CONFIG_FILE', os.path.join(cur_dir, 'img_stats.yaml'))
config = OmegaConf.merge(
    dict(gpus=8, write_to="runner.sh"),
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
    sweep_arg_keys = [x for x in sweep_args.keys()]
    sweep_arg_keys.sort()
    runner_id = f"{model_config}_"
    for k in sweep_arg_keys:
        runner_id += f"{k}_{sweep_args[k]}"
    return runner_id

def main():
    os.makedirs(config.log_dir, exist_ok=True)

    with open(config.write_to, 'w') as f:
        gpu = 0

        for model_idx in range(len(config.models)):
            for sweep_args in get_sweep_args(config.models[model_idx].sweep_args):
                for stats_idx in range(len(config.stats)):
                    model_config = f"models.{model_idx}"
                    stats_config = f"stats.{stats_idx}"

                    runner_id = get_runner_id(model_config, sweep_args)
                    logfile = os.path.join(config.log_dir, runner_id + '.log')

                    envvars = f"CUDA_VISIBLE_DEVICES={gpu}"
                    cmd = f"python {os.path.join(cur_dir, 'generate_images.py')}"
                    args = f"model_config={model_config} stats_config={stats_config} sweep_args=\"{sweep_args}\""

                    run_cmd = f"{envvars} {cmd} {args} &> {logfile} &"

                    f.write(run_cmd + '\n\n')

                    gpu = (gpu + 1) % config.gpus

if __name__ == "__main__":
    main()