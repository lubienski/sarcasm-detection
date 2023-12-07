PYTHONENCODING=utf-8 nohup python run_contBert.py --data_dir ./data/Ghosh \
        --output_dir ./output/Ghosh_ContBert_output_02/ \
	--do_train --do_test --model_select ContBert \
        --know_strategy minor_sent_know.txt 2>&1 & 
