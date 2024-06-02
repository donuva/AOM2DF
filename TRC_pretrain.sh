python3.7 pretrain_trc.py \
      --dataset TRC /home/zhouru/ABSA3/src/data/jsons/TRC_info.json \
      --checkpoint_dir ./checkpoint_dir \
      --model_config config/pretrain_base.json \
      --trc_enabled 1 \
      --log_dir logs \
      --epochs 61 \
      --mrm_loss_type KL \
      --task pretrain \
      --checkpoint ./checkpoint \
      --rank 3