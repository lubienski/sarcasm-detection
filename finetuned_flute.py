import os
from scipy.special import softmax
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from models import Bert, KLBert, Bert_concat
from optimizer import BertAdam
from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    flute = load_dataset("ColumbiaNLP/FLUTE", split = "train").train_test_split(test_size=0.1)
    flute = flute.class_encode_column("label")

    flute_train= flute["train"]
    flute_test= flute["test"]

    output_dir = "flute_here"
    warmup_steps = 5
    num_train_epochs = 8
    learning_rate = 1e-5
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    evaluation_strategy = "epoch"
    disable_tqdm = False
    log_level = "error"
    report_to = "none"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2)
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize(batch):
        raw_encoded = tokenizer(batch["premise"], batch["hypothesis"], padding="max_length", truncation = True)
        return raw_encoded

    flute_encoded_train = flute_train.map(tokenize, batched=True)
    flute_encoded_train = flute_encoded_train.remove_columns(['id', 'hypothesis', 'premise', 'idiom','explanation', 'split', 'type'])
    flute_encoded_train.set_format("torch")

    flute_encoded_test = flute_test.map(tokenize)
    flute_encoded_test = flute_encoded_test.remove_columns(['id', 'hypothesis', 'premise', 'idiom','explanation', 'split', 'type'])
    flute_encoded_test.set_format("torch")

    training_args = TrainingArguments(output_dir = output_dir,\
                        warmup_steps = warmup_steps,\
                        num_train_epochs=num_train_epochs,\
                        learning_rate=learning_rate,\
                        per_device_train_batch_size=per_device_train_batch_size,\
                        per_device_eval_batch_size=per_device_eval_batch_size,\
                        evaluation_strategy=evaluation_strategy,\
                        disable_tqdm=disable_tqdm,\
                        log_level=log_level,\
                        report_to=report_to)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= flute_encoded_train,
        eval_dataset=flute_encoded_test,
        tokenizer=tokenizer
    )
    trainer.train()
    
    torch.save(model, os.path.join(output_dir, "flute_checkpoint.pt"))

    # Evaluation
    trainer.evaluate()
    preds_output = trainer.predict(flute_encoded_test)
    scores = torch.from_numpy(preds_output[0]).softmax(1)
    y_preds = np.argmax(scores.numpy(), axis = 1)
    #accuracy = np.sum(y_preds == dataloader_test["label"])/np.size(y_preds)
    output_file = os.path.join(output_dir, "flute_eval.csv")
    if output_file != None:
        output_df = pd.DataFrame()
        output_df["premise"] = flute_test["premise"]
        output_df["hypothesis"] = flute_test["hypothesis"]
        output_df["true_label"] = flute_encoded_test["label"]
        output_df["pred_label"] = y_preds
        output_df["probit"] = scores.numpy().max(axis = 1)
        output_df.to_csv(output_file, index = False)

    accuracy = accuracy_score(flute_encoded_test["label"], y_preds)
    f1 = f1_score(flute_encoded_test["label"], y_preds, average="weighted", pos_label = 1)
    report_file = os.path.join(output_dir, "flute_report.txt")
    with open(report_file, "w") as writer:
        print("***** Test Eval Restults *****")
        report_str = f"accuracy: {accuracy}, F-1 Score: {f1}"
        print(report_str)
        writer.write(report_str)


if __name__ == "__main__":
    main()
