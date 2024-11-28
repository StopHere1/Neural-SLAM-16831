#！/bin/bash
python frontier_main.py --auto_gpu_config 0 --use_deterministic_local 1 --num_processes 2 --train_global 1 --train_local 0 --train_slam 0
# python main.py --num_episodes 5 --max_episode_length 500 --auto_gpu_config 0 --num_processes 2 --split val_custom --eval 1 --train_global 0 --train_local 0 --train_slam 0 -v 1 \
# --load_global pretrained_models/model_best.global \
# --load_local pretrained_models/model_best.local \
# --load_slam pretrained_models/model_best.slam 