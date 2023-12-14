## CS 769 -  Project Report Repo
### Group 7

Contributors: Yuna Hwang, Anna Lubienski, Jane Zhang


## Requirements
* torch
* transformers
* adapters
* datasets
* tqdm
* scikit-learn
* pandas
* scipy

## Clone the Repo

```
git clone {repo-link}
```

## To Download the model checkpoint
There are two ways of downloading the BERT encoder that we performed task-specific continue pretraining:

1. Go to this Google drive [link](https://drive.google.com/drive/folders/1hZAeXkW35kWDOzSRrRS6N3NflFhdnCAR?usp=sharing), and place the model checkpoint in this folder

2. Or input the following bash command in terminal
it will first install `gdown`, which is a library supports downloading files from google drive.
Then it will use `gdown` to download this checkpoint.
```
bash ./continue_pretrained_bert_ckpt/download_bert_ckpts.sh
```

## Training

The script for training an adapter using FLUTE/Ghosh dataset and the continue-pretrained Bert encoder:

```
PYTHONENCODING=utf-8 python train_adapter.py \
        --dataset_identifier flute \
        --output_dir ./output/adapter_flute_20_epoch_fixed_ContBert_v2\
        --bert_model ./continue_pretrained_bert_ckpt/cont_pretrained_bert.ckpt_3 \
        --learning_rate 2e-4 \
        --num_train_epochs 10
```

or use ```train_adapter.sh```

where 
* `--dataset_identifier` can be set as `flute` or `ghosh` 
* `--output_dir` should keep up with `dataset_identifier` and `num_train_epochs` to be `./output/adapter_[DATASET_IDENTIFIER]_[NUM_TRAIN_EPOCHS]_fixed_ContBert/`
* `--bert_model` path of the Bert encoder we want to use
* `--learning_rate` 
* `--num_train_epochs`

The script for training an adapter using Ghosh dataset and the continue-pretrained Bert encoder:
```
PYTHONENCODING=utf-8 python train_adapter.py \
        --dataset_identifier ghosh \
        --output_dir ./output/adapter_ghosh_20_epoch_fixed_ContBert_v1\
        --bert_model ./continue_pretrained_bert_ckpt/cont_pretrained_bert.ckpt_3 \
        --learning_rate 2e-4 \
        --num_train_epochs 50
```



The script for training with **Adapter Fusion** Layer in the text encoder:
trained adapters are already saved in `./trained_adapters`
```
PYTHONENCODING=utf-8 python run_classifier.py --data_dir ./data/Ghosh \
        --output_dir ./output/FuseAdapterBert_flute_gosh_output_04/ \
	--do_train --do_test --model_select FuseAdapterBert \
        --train_batch_size 32 \
        --num_train_epochs 30 \
        --learning_rate 3e-5 \
        --adapter_task1_dir ./trained_adapters/adapter_flute_20_epoch_fixed_ContBert_v1 \
        --adapter_task2_dir ./trained_adapters/adapter_ghosh_20_epoch_fixed_ContBert_v5 \
        --know_strategy minor_sent_know.txt
```

The script for training with **Stack Adapters** in the text encoder:
```
PYTHONENCODING=utf-8 python run_classifier.py --data_dir ./data/Ghosh \
        --output_dir ./output/StackAdapterBert_flute_gosh_output_02_a2_epoch50/ \
	--do_train --do_test --model_select StackAdapterBert \
        --train_batch_size 32 \
        --num_train_epochs 30 \
        --learning_rate 3e-5 \
        --adapter_task1_dir ./trained_adapters/adapter_flute_20_epoch_fixed_ContBert_v1 \
        --adapter_task2_dir ./trained_adapters/adapter_ghosh_20_epoch_fixed_ContBert_v5 \
        --know_strategy minor_sent_know.txt
```
or can use the script included in `./run_SarDeCK_w_adapter.sh`

(Reported in HW3: Project proposal)The script for using the Continue-pretrained Bert model as the text Encoder to
finetune with SarDeCK using the rest of 90% training data of Ghosh is:
```
PYTHONENCODING=utf-8 python run_contBert.py \
        --data_dir ./data/Ghosh \
        --output_dir ./output/Ghosh_ContBert_output_02/ \
	      --do_train --do_test --model_select ContBert \
        --know_strategy minor_sent_know.txt
```

where 
* `--data_dir` can is set to be `./data/Ghosh` in this project
* `--model_select` can should be `KL-Bert`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`, we chose `minor_sent_know.txt` because this gives the best results empirically according to the paper
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`, by default this is `5` and we use this value because this gives the best results empirically according to the paper




[Code base adapted from https://github.com/LeqsNaN/SarDeCK/tree/main]
