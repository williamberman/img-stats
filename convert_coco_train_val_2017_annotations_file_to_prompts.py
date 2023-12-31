#!/usr/bin/env python

import json
import random
import os

if __name__ == "__main__":
    os.system('wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip')

    os.system('unzip annotations_trainval2017.zip')

    with open("annotations/captions_val2017.json") as f:
        annotations = json.load(f)

    annotations_by_image_id = {}

    for x in annotations["annotations"]:
        if x["image_id"] not in annotations_by_image_id:
            annotations_by_image_id[x["image_id"]] = []
        annotations_by_image_id[x["image_id"]].append(x)

    annotations = [x for x in annotations_by_image_id.values()]

    ctr = 0

    with open('coco-validation-2017-prompts.jsonl', 'w') as f:
        for i, annotation in enumerate(annotations):
            for j in range(2):
                caption = annotation[int(random.random() * len(annotation))]['caption']
                line = json.dumps(caption)
                line += '\n'
                f.write(line)
                ctr += 1

    print(f"wrote total {ctr} prompts")

    os.system('rm -rf annotations annotations_trainval2017.zip')