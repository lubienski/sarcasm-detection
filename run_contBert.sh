PYTHONENCODING=utf-8 python run_contBert.py --data_dir ./data/Ghosh \
        --output_dir ./output/Ghosh_ContBert_output_02/ \
	--do_train --do_test --model_select ContBert \
        --num_train_epochs 8 \
        --know_strategy minor_sent_know.txt
