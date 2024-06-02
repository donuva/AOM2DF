for sl in  '7.5e-5' 
do
		echo ${sl}
		CUDA_VISIBLE_DEVICES=0 python3.7 MAESC_training.py \
          --dataset twitter15 ./src/data/jsons/twitter15_info.json \
          --checkpoint_dir ./train15 \
          --model_config ./Data_New/config/pretrain_base.json \
          --log_dir 15_aesc \
          --num_beams 4 \
          --eval_every 1 \
          --lr ${sl} \
          --batch_size 16  \
          --epochs 35 \
          --grad_clip 5 \
          --warmup 0.1 \
          --seed 57 \
          --checkpoint ./Data_New/checkpoint/pytorch_model.bin \
          --rank 0 \
          --trc_pretrain_file ./Data_New/TRC_ckpt/pytorch_model.bin \
          --nn_attention_on \
          --nn_attention_mode 0\
          --trc_on \
          --gcn_on \
          --dep_mode 2 \
          --sentinet \
          --cpu
done

#python MAESC_training.py --dataset twitter15 ./src/data/jsons/twitter15_info.json --checkpoint_dir ./train15 --model_config ./Data_New/config/pretrain_base.json --log_dir 15_aesc --num_beams 4 --eval_every 1 --lr 7.5e-5 --batch_size 2  --epochs 35 --grad_clip 5 --warmup 0.1 --seed 57 --checkpoint ./Data_New/checkpoint/pytorch_model.bin --rank 0 --trc_pretrain_file ./Data_New/TRC_ckpt/pytorch_model.bin --nn_attention_on --nn_attention_mode 0 --trc_on --gcn_on --dep_mode 2 --sentinet --no_train --cpu
