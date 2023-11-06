GPU='0'


dataset="tiny-imagenet-200"
data_path="./tiny-imagenet-200/"
# dataset='cifar100'
# data_path='./cifar/'
save_dir='./multi_results_run1_tinyImagenet/random_seed_42_Mode_2'

batch_size=400
eval_batch_size=400

epochs=500
lr=0.05
wd=1e-4 # metascorer paper used weight decay of 1e-4
momentum=0.95
k=5
rand_seed=42
meta_lr=1e-5
meta_weight_decay=1e-4

# 1159M MACs, 80.46%
#base_name='ResNet34' 

# 555M MACs, 76.56%
#base_name='ResNet18' 

# 253M MACs, 75.25%
#base_name='ResNet10' 

# 64M MACs, 71.99%
#base_name='ResNet10_l'

# 0.8M MACs, 28.21%
base_name='ResNet10_xxxs'

# 2M MACs, 32.05%
#base_name='ResNet10_xxs'

# 2.86M MACs, 42.99%
#base_name="ResNet10_xs"

# 4M MACs, 52.16%
#base_name="ResNet10_xxxs"
#teacher_name='ResNet34'
teacher_name='ResNet10_l'

# 16M MACs, 65.24%
#base_name='ResNet10_m'

echo "S=$base_name"
echo "T=$teacher_name"
# CUDA_VISIBLE_DEVICES="$GPU"  python train_meta.py --dataset $dataset --data_path $data_path \
#     --teacher $teacher_name --model_name $base_name --save_dir $save_dir \
# 	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
# 	--lr $lr --wd $wd --momentum $momentum --epochs $epochs --meta_lr 0.01 \
# 	--pretrained_student True --temperature 2 --sched_cycles 1 \
# 	--rand_seed $1
CUDA_VISIBLE_DEVICES=$GPU python3 train_multi.py --dataset $dataset --data_path $data_path \
    --teacher $teacher_name --model_name $base_name --save_dir $save_dir \
	--eval_batch_size $eval_batch_size --batch_size $batch_size  \
	--lr $lr --wd $wd --momentum $momentum --epochs $epochs \
	--temperature 2 --rand_seed $rand_seed  --meta_lr $meta_lr --meta_weight_decay $meta_weight_decay #--single True #--correlated True
