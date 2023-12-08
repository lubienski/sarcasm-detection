# PYTHONENCODING=utf-8 nohup python run_classifier.py --data_dir ./data/Ghosh \
#         --output_dir ./output/FuseAdapterBert_flute_gosh_output_05/ \
# 	--do_train --do_test --model_select FuseAdapterBert \
#         --train_batch_size 32 \
#         --num_train_epochs 30 \
#         --learning_rate 3e-5 \
#         --adapter_task1_dir ./trained_adapters/adapter_flute_20_epoch_fixed_ContBert_v1 \
#         --adapter_task2_dir ./trained_adapters/adapter_ghosh_50_epoch_fixed_ContBert_v1 \
#         --know_strategy minor_sent_know.txt > FuseAdapter05.txt 2>&1 &

# StackAdapterBert

PYTHONENCODING=utf-8 nohup python run_classifier.py --data_dir ./data/Ghosh \
        --output_dir ./output/StackAdapterBert_flute_gosh_output_02_a2_epoch50/ \
	--do_train --do_test --model_select StackAdapterBert \
        --train_batch_size 32 \
        --num_train_epochs 30 \
        --learning_rate 3e-5 \
        --adapter_task1_dir ./trained_adapters/adapter_flute_20_epoch_fixed_ContBert_v1 \
        --adapter_task2_dir trained_adapters/adapter_ghosh_50_epoch_fixed_ContBert_v1 \
        --know_strategy minor_sent_know.txt > StackAdapter02.txt 2>&1 &