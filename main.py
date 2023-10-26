# -*- coding: utf-8 -*-

import os,random, argparse
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
from sklearn.metrics import accuracy_score, f1_score
#from transformers import DistilBertConfig

# config = DistilBertConfig()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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
    self.model = AutoModelForSequenceClassification.from_pretrained(config.bert_ckpt,\
                                                            num_labels = config.num_labels,\
                                                            ignore_mismatched_sizes=True)
    self.model = self.model.to(device)
    self.training_args = config.training_args
    self.tokenizer = config.tokenizer


  def train(self, train_dataset, val_dataset, model_name, device):
    self.training_args.output_dir = model_name
    self.trainer = iSarcasmTrainer(
    #self.trainer = Trainer(
        model=self.model,
        args=self.training_args,
        train_dataset= train_dataset,
        eval_dataset=val_dataset,
        tokenizer=self.tokenizer,
    )
    self.trainer.set_device(device)
    self.trainer.train()

  def predict(self, dataloader_test, output_file = None):
    self.trainer.evaluate()
    preds_output = self.trainer.predict(dataloader_test)
    scores = torch.from_numpy(preds_output[0]).softmax(1)
    y_preds = np.argmax(scores.numpy(), axis = 1)
    #accuracy = np.sum(y_preds == dataloader_test["label"])/np.size(y_preds)
    if output_file != None:
        output_df = pd.DataFrame()
        output_df["tweet"] = dataloader_test["tweet"]
        output_df["true_label"] = dataloader_test["label"]
        output_df["pred_label"] = y_preds 
        output_df["probit"] = scores.numpy().max(axis = 1)
        output_df.to_csv(output_file, index = False)

    accuracy = accuracy_score(dataloader_test["label"], y_preds)
    f1 = f1_score(dataloader_test["label"], y_preds, average="weighted", pos_label = 1)
    return (accuracy, f1)

class iSarcasmTrainer(Trainer):
    
    def set_device(self, device):
        self.device = device

    def compute_loss(self, model, inputs, return_outputs = False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fcn = nn.CrossEntropyLoss(weight = torch.tensor([0.75, 0.25], device = self.device))
        #loss_fcn = nn.MultiMarginLoss(p=2)
        loss = loss_fcn(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_dataset(file, rename_dict, tweet_col, label_col):
	df = pd.read_csv(file)
	df = df.rename(columns = rename_dict)
	return Dataset.from_pandas(df[[tweet_col, label_col]])



def main(args):

  default_output_dir = "ckpt"
    
  # Load arguments from args
  bert_ckpt = args.bert_ckpt
  num_labels = args.num_labels
  batch_size = args.batch_size
  weight_decay = args.weight_decay
  warmup_steps = args.warmup
  epochs = args.epochs
  test_label_col = args.test_label_col
  train_label_col = args.train_label_col
  
  TRAIN_FILE = args.train
  TEST_FILE = args.test

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

  tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
  def tokenize(batch):
    return tokenizer(batch["tweet"], padding=True, truncation=True,max_length=512, return_tensors='pt')

  sarcasm_config = Sarcasm_detection_config(num_labels = num_labels,\
                                            tokenizer=tokenizer,\
                                            bert_ckpt = bert_ckpt, \
                                            num_train_epochs = epochs, \
                                            evaluation_strategy = "epoch",\
                                            warmup = warmup_steps, weight_decay = weight_decay, \
                                            per_device_train_batch_size = batch_size, \
                                            per_device_eval_batch_size = 64, \
                                            output_dir = default_output_dir
                                            )
  model_name = f"{bert_ckpt}-finetune-isarcasm"

  # Load datasets
  train_dataset = load_dataset(TRAIN_FILE, \
                                {train_label_col: "label"},
                                "tweet", \
                                "label")
  test_dataset = load_dataset(TEST_FILE,\
                                {test_label_col: "label"},
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
  acc_test, f1_test = sarcasm_detector.predict(test_encoded, output_file = args.test_out)
  print(f"Training: Accuracy: {acc_train}, F1-Score: {f1_train}")
  print(f"Test: Accuracy: {acc_test}, F1-Score: {f1_test}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_ckpt", type=str, default="cardiffnlp/twitter-roberta-base-sentiment")
    parser.add_argument("--train", type=str, default="data/Train_Dataset.csv")
    parser.add_argument("--dev", type=str, default="data/Test_Dataset.csv")
    parser.add_argument("--test", type=str, default="data/Test_Dataset.csv")
    parser.add_argument("--test_label_col", type=str, default = "sarcastic")
    parser.add_argument("--train_label_col", type=str, default = "sarcastic")
    parser.add_argument("--dev_label_col", type=str, default = "sarcastic")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_labels", type=int, default=2)


    parser.add_argument("--dev_out", type=str, default="isarcasm-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="isarcasm-test-output.txt")
    parser.add_argument("--filepath", type=str, default=None)
    parser.add_argument("--warmup", type=float,default=0)


    # hyper parameters
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
  args = get_args()
  if args.filepath is None:
        args.filepath = f'{args.epochs}-{args.lr}.pt' # save path
  seed_everything(args.seed)
  main(args)


