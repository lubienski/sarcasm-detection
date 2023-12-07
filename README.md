## CS 769 -  Project Proposal Repo
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

  # Follow instruction in ./continue_pretrained_bert_ckpt/README.md to download the trained `task-specific continue-pretrained BERT encoder` and place it in ./continue_pretrained_bert_ckpt

## Training
The script for using 10% Ghosh Data for Continue-pretraining is:
```
# TODO
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`


The script for using the Continue-pretrained Bert model as the text Encoder to
finetune with SarDeCK using the rest of 90% training data of Ghosh is:
```
# TODO
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`

The script for training an adapter using FLUTE dataset and the continue-pretrained Bert encoder:
```
# TODO
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`

The script for training an adapter using Ghosh dataset and the continue-pretrained Bert encoder:
```
# TODO
```
where 


The script for training with Adapter Fusion Layer in the text encoder:
```
# TODO
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`

  The script for training with Stack Adapters in the text encoder:
```
# TODO
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`  
* `--model_select` can be set as `KL-Bert`, `Bert_concat`, and `Bert-Base`  
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, and `minor_sent_know.txt`
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`


[Code base adapted from https://github.com/LeqsNaN/SarDeCK/tree/main]
