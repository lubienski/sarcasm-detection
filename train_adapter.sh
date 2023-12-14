PYTHONENCODING=utf-8 python train_adapter.py \
        --dataset_identifier flute \
        --output_dir ./output/adapter_flute_20_epoch_fixed_ContBert_v2\
        --bert_model ./cont_pretrained_bert.ckpt_3 \
        --learning_rate 2e-4 \
        --num_train_epochs 50

PYTHONENCODING=utf-8 python train_adapter.py \
        --dataset_identifier ghosh \
        --output_dir ./output/adapter_ghosh_20_epoch_fixed_ContBert_v1\
        --bert_model ./cont_pretrained_bert.ckpt_3 \
        --learning_rate 3e-4 \
        --num_train_epochs 50

# PYTHONENCODING=utf-8 nohup python train_adapter.py \
#         --dataset_identifier ghosh \
#         --output_dir ./output/adapter_ghosh_20_epoch_fixed_ContBert_v5\
#         --bert_model ./cont_pretrained_bert.ckpt_3 \
#         --learning_rate 2e-4 \
#         --dataset_identifier ghosh \
#         --num_train_epochs 20  > adapter_ghosh_fixed_bert_v5.txt 2>&1 &