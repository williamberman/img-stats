import argparse
import logging
import json
import torch
from transformers import CLIPModel, CLIPProcessor
import os
from PIL import Image

logger = logging.getLogger(__name__)

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--path', required=True)
    args.add_argument("--device", required=False, default="cuda", type=str)
    args = args.parse_args()

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    clip.to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    clip_scores_sum = 0
    num_images = 0

    with open(os.path.join(args.path, 'index.json')) as f:
        while True:
            line = f.readline()
            if line == '':
                break
            filename, text = json.loads(line)
            filepath = os.path.join(args.path, filename + '.png')
            generated_image = Image.open(filepath)

            input = clip_processor(
                text=text,
                images=generated_image,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            )
            input["pixel_values"] = input["pixel_values"].to(dtype=torch.float16, device=args.device)
            input["input_ids"] = input["input_ids"].to(args.device)
            input["attention_mask"] = input["attention_mask"].to(args.device)

            clip_scores = clip(**input).logits_per_image.diag()
            clip_scores_sum += clip_scores.sum().item()
            num_images += len(clip_scores)

            logger.warning(f"done: {num_images} current clip score {clip_scores_sum / num_images}")

    logger.warning(f"final: done: {num_images}, {clip_scores_sum / num_images}")

if __name__ == "__main__":
    main()