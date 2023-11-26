# CUDA_VISIBLE_DEVICES=0 python3 ./src/main.py --ensemble_size 1 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity
CUDA_VISIBLE_DEVICES=3 python3 ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity
CUDA_VISIBLE_DEVICES=3 python3 ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001 --no_diversity


# DBAT: D_ood = D_test
# CUDA_VISIBLE_DEVICES=0 python3 ./src/main.py --ensemble_size 1 --batch_size_train 128 --batch_size_eval 512 --seed 0 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
CUDA_VISIBLE_DEVICES=3 python3 ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 1 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
CUDA_VISIBLE_DEVICES=3 python3 ./src/main.py --ensemble_size 5 --batch_size_train 128 --batch_size_eval 512 --seed 2 --epochs 300 --lr 0.001 --opt sgd --scheduler none --alpha 0.0001 --perturb_type ood_is_test --model resnet18 --dataset waterbird --eval_freq 1000 --l2_reg 0.0001
