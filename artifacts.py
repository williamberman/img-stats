#!/usr/bin/env python

import os
import json
import logging
import torch
from omegaconf import OmegaConf
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver('torch_dtype', lambda x: getattr(torch, x))
OmegaConf.register_new_resolver(
    'wurst_stage_c_timesteps', lambda: DEFAULT_STAGE_C_TIMESTEPS)

logger = logging.getLogger(__name__)

cur_dir = os.path.dirname(os.path.abspath(__file__))

cli_config = OmegaConf.merge(dict(config_file=os.path.join(cur_dir, "img_stats.yaml")), OmegaConf.from_cli())
config = OmegaConf.merge(
    dict(only=None, skip={}, skip_guidance_scales=[], save_to_root=False),
    OmegaConf.load(cli_config.config_file), 
    cli_config,
)

if isinstance(config.skip_guidance_scales, str):
    config.skip_guidance_scales = [float(x) for x in config.skip_guidance_scales.split(',')]
if isinstance(config.skip_guidance_scales, (int, float)):
    config.skip_guidance_scales = [float(config.skip_guidance_scales)]

if config.only is not None:
    if isinstance(config.only, str):
        model_indices = [int(x) for x in config.only.split(',')]
    elif isinstance(config.only, (int, float, str)):
        model_indices = [int(config.only)]
    else:
        model_indices = config.only
else:
    model_indices = [x for x in range(len(config.models))]

if isinstance(config.skip, str):
    config.skip = json.loads(config.skip)

if config.save_to_root:
    os.makedirs(os.path.join(cur_dir, 'charts'), exist_ok=True)

def main():
    for stats_idx in range(len(config.stats)):
        metrics = config.stats[stats_idx].metrics

        if "clip" in metrics:
            make_clip_chart()
            clip_table = make_clip_table()
        else:
            clip_table = None

        if "fid" in metrics:
            make_fid_chart()
            fid_table = make_fid_table()
        else:
            fid_table = None

        if "isc" in metrics:
            make_isc_chart()
            isc_table = make_isc_table()
        else:
            isc_table = None

        if "fid" in metrics and "clip" in metrics:
            make_fid_vs_clip_chart()

        with open(os.path.join(cur_dir, config.save_to.path, config.run_prefix + '_tables.txt'), 'w') as tables:
            tables.write(combine_tables(clip_table, fid_table, isc_table))

def iter_helper():
    rv = []

    for model_idx in model_indices:
        if "sweep_args" in config.models[model_idx] and "guidance_scale" in config.models[model_idx].sweep_args:
            guidance_scales = [x for x in config.models[model_idx].sweep_args.guidance_scale if x not in config.skip_guidance_scales]
        elif "args" in config.models[model_idx] and "prior_guidance_scale" in config.models[model_idx].args:
            guidance_scales = [config.models[model_idx].args.prior_guidance_scale]
        else:
            continue

        sweep_args = dict(
            config.models[model_idx].get('sweep_args', {}))
        sweep_args.pop("guidance_scale", None)
        sweep_args = get_sweep_args(sweep_args)

        for sweep_args in sweep_args:
            if model_idx in config.skip:
                skip_sweep = False
                for skip_sweep_arg_k, skip_sweep_arg_v in config.skip[model_idx].items():
                    if sweep_args[skip_sweep_arg_k] == skip_sweep_arg_v:
                        skip_sweep = True
                if skip_sweep:
                    logger.warning('skipping sweep')
                    continue

            clip_scores = []
            fid_scores = []
            isc_scores = []

            for guidance_scale in guidance_scales:
                full_sweep_args = dict(sweep_args)

                if "sweep_args" in config.models[model_idx] and "guidance_scale" in config.models[model_idx].sweep_args:
                    full_sweep_args['guidance_scale'] = guidance_scale

                prefixes = [f"{config.run_prefix}_models.{model_idx}"]

                if 'prefix_override' in config.models[model_idx]:
                    prefixes.append(config.models[model_idx].prefix_override)

                found_metrics = {}

                for run_prefix in prefixes:
                    if len(full_sweep_args) > 0:
                        runner_id = f"{run_prefix}_{serialize_sweep_args(full_sweep_args)}"
                    else:
                        runner_id = run_prefix

                    assert config.save_to.type == 'local_fs'
                    run_save_path = os.path.join(
                        cur_dir, config.save_to.path, runner_id)
                    filename = os.path.join(run_save_path, 'metrics.json')

                    if os.path.exists(filename):
                        with open(filename) as f:
                            found_metrics = json.load(f)
                        break

                clip_scores.append(found_metrics.get("clip", None))
                fid_scores.append(found_metrics.get("frechet_inception_distance", None))
                isc_scores.append(found_metrics.get("inception_score_mean", None))

            full_sweep_args = dict(sweep_args)

            if 'num_inference_steps' in full_sweep_args:
                num_inference_steps = full_sweep_args["num_inference_steps"]
            else:
                if config.models[model_idx].name == "wuerstchen":
                    num_inference_steps = 41
                else:
                    num_inference_steps = config.models[model_idx].args.num_inference_steps
                full_sweep_args['num_inference_steps'] = num_inference_steps

            label = f"{config.models[model_idx].name}_{serialize_sweep_args(full_sweep_args)}"

            rv.append(dict(
                guidance_scales=guidance_scales, clip_scores=clip_scores, 
                fid_scores=fid_scores, isc_scores=isc_scores, 
                label=label, 
                model_config=config.models[model_idx],
                max_clip_score=max([x for x in clip_scores if x is not None]),
                min_fid_score=min([x for x in fid_scores if x is not None]),
                max_isc_score=max([x for x in isc_scores if x is not None]),
                num_inference_steps=num_inference_steps,
            ))

    return rv

def make_clip_chart():
    logger.warning(f"making clip chart")

    plt.figure()
    plt.grid(True)
    plt.title(f"CLIP Score")
    plt.ylabel("CLIP Score (10k)")
    plt.xlabel("cfg scale")

    for it in iter_helper():
        plt.plot(it["guidance_scales"], it["clip_scores"], marker="o", label=it["label"])

    plt.savefig(os.path.join(cur_dir, config.save_to.path, config.run_prefix + '_clip.png'))
    if config.save_to_root:
        plt.savefig(os.path.join(cur_dir, 'charts', 'clip.png'))
    plt.close()

def make_clip_table():
    logger.warning("making clip table")

    table = r"""\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{ } & \textbf{clip} & \textbf{guidance scale} & \textbf{timesteps} & \textbf{resolution} \\ \hline
"""

    iter_over = iter_helper()

    iter_over.sort(key=lambda it: it["max_clip_score"], reverse=True)

    for it in iter_over:
        table += r"\textbf{" + it["model_config"].name + "}" + f" & {round(it['max_clip_score'], 2)} & {it['guidance_scales'][it['clip_scores'].index(it['max_clip_score'])]} & {it['num_inference_steps']} & {it['model_config'].resolution}" + r" \\ \hline" + "\n"

    table += r"\end{tabular}" + "\n"

    return table

def make_fid_chart():
    logger.warning(f"making fid chart")

    plt.figure()
    plt.grid(True)
    plt.title(f"FID")
    plt.ylabel("FID Score (10k)")
    plt.xlabel("cfg scale")

    for it in iter_helper():
        plt.plot(it["guidance_scales"], it["fid_scores"], marker="o", label=it["label"])

    plt.savefig(os.path.join(cur_dir, config.save_to.path, config.run_prefix + '_fid.png'))
    if config.save_to_root:
        plt.savefig(os.path.join(cur_dir, 'charts', 'fid.png'))
    plt.close()

def make_fid_table():
    logger.warning("making fid table")

    table = r"""\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{ } & \textbf{fid} & \textbf{guidance scale} & \textbf{timesteps} & \textbf{resolution} \\ \hline
"""

    iter_over = iter_helper()

    iter_over.sort(key=lambda it: it["min_fid_score"])

    for it in iter_over:
        table += r"\textbf{" + it["model_config"].name + "}" + f" & {round(it['min_fid_score'], 2)} & {it['guidance_scales'][it['fid_scores'].index(it['min_fid_score'])]} & {it['num_inference_steps']} & {it['model_config'].resolution}" + r" \\ \hline" + "\n"

    table += r"\end{tabular}" + "\n"

    return table

def make_isc_chart():
    logger.warning(f"making fid chart")

    plt.figure()
    plt.grid(True)
    plt.title(f"Inception Score")
    plt.ylabel("Inception Score (10k)")
    plt.xlabel("cfg scale")

    for it in iter_helper():
        plt.plot(it["guidance_scales"], it["isc_scores"], marker="o", label=it["label"])

    plt.savefig(os.path.join(cur_dir, config.save_to.path, config.run_prefix + '_isc.png'))
    if config.save_to_root:
        plt.savefig(os.path.join(cur_dir, 'charts', 'isc.png'))
    plt.close()

def make_isc_table():
    logger.warning("making isc table")

    table = r"""\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{ } & \textbf{isc} & \textbf{guidance scale} & \textbf{timesteps} & \textbf{resolution} \\ \hline
"""

    iter_over = iter_helper()

    iter_over.sort(key=lambda it: it["max_isc_score"], reverse=True)

    for it in iter_over:
        table += r"\textbf{" + it["model_config"].name + "}" + f" & {round(it['max_isc_score'], 2)} & {it['guidance_scales'][it['isc_scores'].index(it['max_isc_score'])]} & {it['num_inference_steps']} & {it['model_config'].resolution}" + r" \\ \hline" + "\n"

    table += r"\end{tabular}" + "\n"

    return table

def make_fid_vs_clip_chart():
    logger.warning(f"making fid vs clip chart")

    plt.figure()
    plt.grid(True)
    plt.title(f"FID vs CLIP")
    plt.ylabel("FID Score (10k)")
    plt.xlabel("CLIP Score (10k)")

    for it in iter_helper():
        plt.plot(it["clip_scores"], it["fid_scores"], marker="o", label=it["label"])

    plt.legend()
    plt.savefig(os.path.join(cur_dir, config.save_to.path, config.run_prefix + '_fid_vs_clip.png'))
    if config.save_to_root:
        plt.savefig(os.path.join(cur_dir, 'charts', 'fid_vs_clip.png'))
    plt.close()

def combine_tables(clip_table, fid_table, isc_table):
    assert clip_table is not None and fid_table is not None and isc_table is not None
    combined = r"\begin{minipage}{0.5\textwidth}" + "\n" + r"\resizebox{\textwidth}{!}{%" + "\n" + clip_table + "}" + "\n" + r"\end{minipage}%" + "\n"
    combined += r"\begin{minipage}{0.5\textwidth}" + "\n" + r"\resizebox{\textwidth}{!}{%" + "\n" + fid_table + "}" + "\n" + r"\end{minipage}%" + "\n\n"
    combined += r"\begin{center}" + "\n" r"\begin{minipage}{0.5\textwidth}" + "\n" + r"\resizebox{\textwidth}{!}{%" + "\n" + isc_table + "}" + "\n" + r"\end{minipage}%" + "\n" + r"\end{center}" + "\n"
    return combined

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

def serialize_sweep_args(sweep_args):
    sweep_arg_keys = [x for x in sweep_args.keys()]
    sweep_arg_keys.sort()

    run_suffix = ''

    for k in sweep_arg_keys:
        if run_suffix == '':
            run_suffix += f"{k}_{sweep_args[k]}"
        else:
            run_suffix += f"_{k}_{sweep_args[k]}"

    return run_suffix

if __name__ == "__main__":
    main()
