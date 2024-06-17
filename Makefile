demo_small:
	CUDA_VISIBLE_DEVICES=0 python demo_2x.py --model ours_small
demo_big:
	CUDA_VISIBLE_DEVICES=0 python demo_2x.py --model ours
demo_quad:
	CUDA_VISIBLE_DEVICES=1 python demo_2x.py --model ours_small_quad
vimeo:
	CUDA_VISIBLE_DEVICES=1 python benchmark/Vimeo90K.py --model ours_small_quad

train:
	  CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port=1123 --nproc_per_node=1 train.py   --model small --world_size 1 --batch_size 32 --data_path /data/dataset/vimeo_dataset/vimeo_triplet --dataset vimeo_triplet


train_nattn:
	  CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port=1235 --nproc_per_node=1 train.py   --model nattn --world_size 1 --batch_size 32 --data_path /data/dataset/vimeo_dataset/vimeo_triplet --dataset vimeo_triplet


train_setuplet:
	  python -m torch.distributed.launch --master_port=1234 --nproc_per_node=1 train.py  --model my_t --world_size 1 --batch_size 32 --data_path /data/dataset/vimeo_dataset/vimeo_setuplet --dataset vimeo_setuplet
train_triplet:
	  python -m torch.distributed.launch --nproc_per_node=1 train.py  --model small --world_size 1 --batch_size 32 --data_path /data/dataset/vimeo_dataset/vimeo_triplet --dataset vimeo_triplet
train_flow:
	  python -m torch.distributed.launch --nproc_per_node=1 train_flow.py   --model my_flow_v2 --world_size 1 --batch_size 32 --data_path /data/dataset/vimeo_dataset/vimeo_triplet --dataset vimeo_triplet_flow_bi

#wandb run --name "실험 이름" python train.py
train_r:
	  python -m torch.distributed.launch  --master_port=1234 --nproc_per_node=1 train.py  --world_size 1 --batch_size 32    --model my  --resume my_tt_297_31.86 --data_path /data/dataset/vimeo_dataset/vimeo_setuplet --dataset vimeo_setuplet

bench:
	python model_benchmark.py --model my_t --resume my_tt_299_31.79  

bench_opt:
	python model_benchmark_opt.py --model my_t --resume my_tt_231_31.77 --bench XTest_8X

eval:
	python run.py --first images/img1.png --second images/img2.png --output images/out.png