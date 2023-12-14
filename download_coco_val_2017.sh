#! /bin/bash

set -e

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017
mv val2017 coco_val_2017
rm val2017.zip