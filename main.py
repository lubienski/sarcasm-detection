# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import DistilBertTokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

from datasets import Dataset
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import DistilBertConfig

# config = DistilBertConfig()

class Sarcasm_detection_config():
  def __init__(self, num_labels, tokenizer, bert_ckpt, per_device_train_batch_size, \
               per_device_eval_batch_size, output_dir,\
               warmup = 0, \
               option = "pretrain", num_train_epochs = 10, \
              learning_rate=2e-5,\
              weight_decay=0.01, evaluation_strategy="epoch",\
              disable_tqdm=False, log_level="error", report_to="none"):
    self.bert_ckpt = bert_ckpt
    self.option = option #Option = "pretrain" or "finetune"
    self.num_labels = num_labels
    self.tokenizer = tokenizer
    self.training_args = TrainingArguments(output_dir = output_dir,\
                        warmup_steps = warmup,\
                        num_train_epochs=num_train_epochs,\
                        learning_rate=learning_rate,\
                        per_device_train_batch_size=per_device_train_batch_size,\
                        per_device_eval_batch_size=per_device_eval_batch_size,\
                        weight_decay=weight_decay,\
                        evaluation_strategy=evaluation_strategy,\
                        disable_tqdm=disable_tqdm,\
                        log_level=log_level,\
                        report_to=report_to)


class Sarcasm_detection():

  def __init__(self, config, device):
    self.model = AutoModelForSequenceClassification.from_pretrained(config.bert_ckpt, num_labels = config.num_labels, ignore_mismatched_sizes=True)
    self.model = self.model.to(device)
    self.training_args = config.training_args
    self.tokenizer = config.tokenizer


  def train(self, train_dataset, val_dataset, model_name, device):
    self.training_args.output_dir = model_name
    self.trainer = Trainer(
        model=self.model,
        args=self.training_args,
        train_dataset= train_dataset,
        eval_dataset=val_dataset,
        tokenizer=self.tokenizer,
    )

    self.trainer.train()

  def predict(self, dataloader_test):
    self.trainer.evaluate()
    preds_output = self.trainer.predict(dataloader_test)
    y_preds = np.argmax(preds_output.predictions, axis = 1)
    accuracy = np.sum(y_preds == dataloader_test["label"])/np.size(y_preds)
    f1 = f1_score(dataloader_test["label"], y_preds, average="weighted")
    return (accuracy, f1)
    

def load_dataset(file, rename_dict, tweet_col, label_col):
	df = pd.read_csv(file)
	df = df.rename(columns = rename_dict)
	return Dataset.from_pandas(df[[tweet_col, label_col]])

def main():
  bert_ckpt = "cardiffnlp/twitter-roberta-base-sentiment"
  num_labels = 2
  batch_size = 32
  weight_decay = 0.01
  warmup_steps = 500
  epochs = 5

  default_output_dir = "ckpt"
  DATA_DIR = "data"
  TRAIN_FILE = "isarcasm_train_v2.csv"
  TEST_FILE = "isarcasm_test_v2.csv"

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

  tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
  def tokenize(batch):
    return tokenizer(batch["tweet"], padding=True, truncation=True,max_length=512, return_tensors='pt')

  sarcasm_config = Sarcasm_detection_config(num_labels = num_labels,\
                                            tokenizer=tokenizer,\
                                            bert_ckpt = bert_ckpt, \
                                            num_train_epochs = epochs, \
                                            evaluation_strategy = "steps",\
                                            warmup = warmup_steps, weight_decay = weight_decay, \
                                            per_device_train_batch_size = batch_size, \
                                            per_device_eval_batch_size = batch_size*2, \
                                            output_dir = default_output_dir
                                            )
  model_name = f"{bert_ckpt}-finetune-isarcasm"

  # Load datasets
  train_dataset = load_dataset(os.path.join(DATA_DIR, TRAIN_FILE), \
                                {"sarcastic": "label"},
                                "tweet", \
                                "label")
  test_dataset = load_dataset(os.path.join(DATA_DIR, TEST_FILE),\
                                {"sarcasm": "label"},
                                "tweet", \
                                "label")
  train_dataset = train_dataset.class_encode_column("label")
  test_dataset = test_dataset.class_encode_column("label")

  train_encoded = train_dataset.map(tokenize, batched = True) 
  test_encoded = test_dataset.map(tokenize, batched = True)

#  train_input_ids = tokenizer(train_dataset["tweet"], truncation=True, padding=True, return_tensors='pt')
 # test_input_ids = tokenizer(test_dataset["tweet"], truncation=True, padding=True, return_tensors='pt')
  
#  train_encoded = SarcasmDataset(train_input_ids, train_dataset["label"])
#  test_encoded = SarcasmDataset(test_input_ids, test_dataset["label"])
#  train_encoded.set_format(type = "torch", columns=["label", "input_ids", "attention_mask"])
 # test_encoded.set_format(type = "torch", columns=["label", "input_ids", "attention_mask"])
  #
  sarcasm_detector = Sarcasm_detection(sarcasm_config, device)
  sarcasm_detector.train(train_encoded, test_encoded, model_name, device)
  acc_train, f1_train =  sarcasm_detector.predict(train_encoded)
  acc_test, f1_test = sarcasm_detector.predict(test_encoded)
  print(f"Training: Accuracy: {acc_train}, F1-Score: {f1_train}")
  print(f"Test: Accuracy: {acc_test}, F1-Score: {f1_test}")

if __name__ == "__main__":
  main()


