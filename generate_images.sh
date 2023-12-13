export SAVE_PATH=~/out/muse_512
export CAPTIONS_FILE=~/annotations/captions_train2017.json
export MODEL=muse-512

time python generate_images.py --guidance_scale 1.5 &> out-1.5 &
time python generate_images.py --guidance_scale 2 &> out-2 &
time python generate_images.py --guidance_scale 3 &> out-3 &
time python generate_images.py --guidance_scale 4 &> out-4 &
time python generate_images.py --guidance_scale 5 &> out-5 &
# time python generate_images.py --guidance_scale 6 &> out-6 &
# time python generate_images.py --guidance_scale 7 &> out-7 &
# time python generate_images.py --guidance_scale 8 &> out-8 &
# time python generate_images.py --guidance_scale 10 &> out-10 &
# time python generate_images.py --guidance_scale 15 &> out-15 &
# time python generate_images.py --guidance_scale 20 &> out-20 &
