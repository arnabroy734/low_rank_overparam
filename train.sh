# tmux new -s normal
# export CUDA_VISIBLE_DEVICES=0
# python main.py --dataset_name chestexpert

# tmux new -s overparam_2_fc
# export CUDA_VISIBLE_DEVICES=1
# python main.py --dataset_name chestexpert --overparam --layers fc --depth 2

# tmux new -s overparam_2_conv
# export CUDA_VISIBLE_DEVICES=2
# python main.py --dataset_name chestexpert --overparam --layers conv --depth 2

# tmux new -s overparam_2_all
# export CUDA_VISIBLE_DEVICES=3
# python main.py --dataset_name chestexpert --overparam --layers all --depth 2

# tmux new -s overparam_4_fc
# export CUDA_VISIBLE_DEVICES=4
# python main.py --dataset_name chestexpert --overparam --layers fc --depth 4

# tmux new -s overparam_4_conv
# export CUDA_VISIBLE_DEVICES=5
# python main.py --dataset_name chestexpert --overparam --layers conv --depth 4


# tmux new -s overparam_4_all
# export CUDA_VISIBLE_DEVICES=6
# python main.py --dataset_name chestexpert --overparam --layers all --depth 4

