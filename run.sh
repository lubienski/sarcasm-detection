#PID= `3`

#OUTPUT_DIR = `output`
#current_date_time=`date + '%Y-%m-%d %T'`


#python main.py --warmup 500 \
		#--train 'data/Train_Dataset.csv' \
		#--test 'data/Test_Dataset.csv'  \
		#--dev 'data/Test_Dataset.csv'

python main.py --warmup 500 \
		--train 'data/isarcasm_train_v2.csv'\
	       	--test 'data/isarcasm_test_v2.csv' \
		--test_label_col "sarcasm" \
	       	--dev 'data/isarcasm_test_v2.csv'\
	       	--dev_label_col "sarcasm"\
	       	--test_out 'isarcasm_original_test_output.csv'
#python main.py  > ./output/output_${current_date_time}_${PID}.txt
