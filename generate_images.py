import os
import argparse
import json
import logging
import torch
from diffusers import DiffusionPipeline
import uuid

logger = logging.getLogger(__name__)

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--slurm", action="store_true")
    args.add_argument("--device", type=str, default="cuda", required=False)
    args.add_argument("--guidance_scale", type=float, required=True)
    args.add_argument("--save_path", type=str, required=False)
    args.add_argument("--batch_size", type=int, required=False, default=4)
    args.add_argument("--model", choices=["sd15", "muse-256", "muse-512"], required=False)
    # http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    args.add_argument("--captions_file", required=False)
    args = args.parse_args()

    if args.save_path is None:
        args.save_path = os.environ["SAVE_PATH"]

    if args.captions_file is None:
        args.captions_file = os.environ["CAPTIONS_FILE"]

    if args.model is None:
        args.model = os.environ["MODEL"]

    args.save_path = os.path.join(args.save_path, str(args.guidance_scale))

    os.makedirs(args.save_path, exist_ok=True)

    if args.slurm:
        ntasks = int(os.environ["SLURM_NTASKS"])
        procid = int(os.environ["SLURM_PROCID"])
    else:
        ntasks = 1
        procid = 0

    with open(args.captions_file) as f:
        annotations = json.load(f)

    annotations_by_image_id = {}

    for x in annotations["annotations"]:
        if x["image_id"] not in annotations_by_image_id:
            annotations_by_image_id[x["image_id"]] = []
        annotations_by_image_id[x["image_id"]].append(x)

    annotations = [x for x in annotations_by_image_id.values()]
    
    logger.warning(f"num images {len(annotations)}.")
    logger.warning("for validation 2017, this should be 5000")

    if args.model == "sd15":
        pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", safety_checker=None).to(args.device)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(args.device)
    elif args.model == "muse-512":
        pipeline = DiffusionPipeline.from_pretrained("huggingface/amused-512", torch_dtype=torch.float16, variant="fp16").to(args.device)
        pipeline.vqvae.to(torch.float32)
        pipeline.set_progress_bar_config(disable=True)
    elif args.model == "muse-256":
        pipeline = DiffusionPipeline.from_pretrained("huggingface/amused-256", torch_dtype=torch.float16, variant="fp16").to(args.device)
        pipeline.vqvae.to(torch.float32)
        pipeline.set_progress_bar_config(disable=True)
    else:
        assert False

    prompts_per_task = len(annotations) // ntasks

    start_prompt = prompts_per_task * procid

    generator = torch.Generator(args.device).manual_seed(0)

    with open(os.path.join(args.save_path, "index.json"), "w") as index_file:
        for prompt_idx in range(start_prompt, start_prompt+prompts_per_task, args.batch_size):
            prompts = [
                annotations[prompt_idx_][0]["caption"]
                for
                prompt_idx_ in range(prompt_idx, prompt_idx+args.batch_size)
            ]
            logger.warning(prompt_idx)

            images = pipeline(
                prompts,
                generator=generator,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=2 # 2 per for 5k prompts brings us to 10k images
            ).images

            for image_idx, image in enumerate(images):
                filename = str(uuid.uuid4())
                filepath = os.path.join(args.save_path, f"{filename}.png")
                image.save(filepath)
                prompt = prompts[image_idx % len(prompts)]
                index_file.write(json.dumps([filename, prompt]) + '\n')

if __name__ == "__main__":
    main()