import numpy as np
import os, argparse
from datasets import load_dataset
from transformers import TrainingArguments, EvalPrediction
from adapters import AdapterTrainer,BertAdapterModel
from transformers import BertConfig
from transformers import BertTokenizer
import torch
import pandas as pd
from datasets import Features,ClassLabel, Value

"""
This file train a adapter for either the flute dataset
or the ghosh dataset

IMPORTANT: it freeze all parameters in the BERT model during training
"""

FLUTE_IDENTIFIER = "flute"
GHOSH_IDENTIFIER = "ghosh"
CONT_BERT_PATH = "./cont_pretrained_bert.ckpt_3"
TASK1 = "contradict"
TASK2 = "sarc"

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}


def get_ghosh_encoded_dataset():
    """
    Load training, dev, and test dataset for Ghosh dataset
    Return training, dev, and test dataset for Ghosh dataset, and id2label
    """
    ft = Features({'row_idx': Value('string'),\
               'text': Value('string'), \
               'labels': ClassLabel(names=["non-sarcastic", "sarcastic"])})
    
    # ghosh_adapter_training_data_path = "./data/Ghosh/15_percent_train_data.txt"
    # ghosh_adapter_training_data_path = "./data/Ghosh/first_50_percent_train_data.txt"
    ghosh_adapter_training_data_path = "./data/Ghosh/train.txt"

    print(f"{ghosh_adapter_training_data_path}")

    ghosh_train = load_dataset("csv",
                column_names = ["row_idx", "text", "labels"],
                data_files=[ghosh_adapter_training_data_path],
                delimiter="==sep==",
                features= ft)

    ghosh_train = ghosh_train.remove_columns(["row_idx"])["train"]

    id2label = {id: label for (id, label) in enumerate(ghosh_train.features["labels"].names)}


    ghosh_dev = load_dataset("csv",
                column_names = ["row_idx", "text", "labels"],
                data_files=["./data/Ghosh/dev.txt"],
                delimiter="==sep==",
                features= ft)

    ghosh_dev = ghosh_dev.remove_columns(["row_idx"])["train"]


    ghosh_test = load_dataset("csv",
                    column_names = ["row_idx", "text", "labels"],
                    data_files=["./data/Ghosh/test.txt"],
                    delimiter="==sep==",
                    features= ft)

    ghosh_test = ghosh_test.remove_columns(["row_idx"])["train"]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_ghosh(batch):
        raw_encoded = tokenizer(batch["text"], padding="max_length", truncation = True)
        return raw_encoded

    ghosh_train_encoded = ghosh_train.map(tokenize_ghosh,batched=True)
    ghosh_train_encoded = ghosh_train_encoded.remove_columns(['text'])
    ghosh_train_encoded.set_format("torch")

    ghosh_dev_encoded = ghosh_dev.map(tokenize_ghosh,batched=True)
    ghosh_dev_encoded = ghosh_dev_encoded.remove_columns(['text'])
    ghosh_dev_encoded.set_format("torch")

    ghosh_test_encoded = ghosh_test.map(tokenize_ghosh,batched=True)
    ghosh_test_encoded = ghosh_test_encoded.remove_columns(['text'])
    ghosh_test_encoded.set_format("torch")

    return ghosh_train_encoded, ghosh_dev_encoded, ghosh_test_encoded,id2label

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_identifier",
                        default=FLUTE_IDENTIFIER,
                        type=str,
                        help="choose from ['flute', 'ghosh]")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                            "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                            "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=40,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--logging_steps",
                        default=500,
                        type=int,
                        help="Log training loss every k steps")
    return parser.parse_args()

def main(args):
    print(f"************* Adapter for {args.dataset_identifier} will be trained *************")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def encode_batch_flute(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            max_length=180,
            truncation=True,
            padding="max_length"
        )

    if args.dataset_identifier == FLUTE_IDENTIFIER:
        flute = load_dataset("ColumbiaNLP/FLUTE", split = "train").train_test_split(test_size=0.1)
        flute = flute.class_encode_column("label")

        # Encode the input data
        flute_ds = flute.map(encode_batch_flute, batched=True)
        # The transformers model expects the target class column to be named "labels"
        flute_ds = flute_ds.rename_column("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        flute_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        id2label = {id: label for (id, label) in enumerate(flute["train"].features["label"].names)}

        config = BertConfig.from_pretrained(
            "bert-base-uncased",
            id2label=id2label,
        )
        model = BertAdapterModel.from_pretrained(
            "bert-base-uncased",
            config=config,
        )
        if args.bert_model == CONT_BERT_PATH:
            print(f"Loading from pretrained Encoder from {args.bert_model}")
            contBert_saved_info = torch.load("./cont_pretrained_bert.ckpt_3", map_location=torch.device('cpu'))
            contBert_state_dict = contBert_saved_info["model_state_dict"]
            # contBert_state_dict['bert.embeddings.position_ids'] = model.state_dict()["bert.embeddings.position_ids"]
            model.load_state_dict(contBert_state_dict, strict=False)
        model.add_adapter(TASK1)
        model.add_classification_head(
            TASK1,
            num_labels = 2,
            id2label=id2label
        )
        model.train_adapter(TASK1)
        print(model.adapter_summary())


        training_args = TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset= flute_ds["train"],
            eval_dataset= flute_ds["test"],
            compute_metrics=compute_accuracy,
        )

        trainer.train()
        print("***** End of traing *****")
        print("***** Training dataset results *****")
        print(trainer.evaluate(flute_ds["train"]))
        print("***** Eval dataset results *****")
        print(trainer.evaluate())

        # Save the model and adapter
        model_dir = os.path.join(args.output_dir, "trained_model_after_adapter")
        os.mkdir(model_dir)
        model_dir = os.path.join(args.output_dir, "trained_model_after_adapter", "model.ckpt")
        # Save model
        torch.save(model.state_dict(), model_dir)

        # Save adapter
        adapter_dir = os.path.join(args.output_dir, "adapter_ckpt")
        model.save_adapter(adapter_dir, TASK1)

        print(f"learning_rate= {args.learning_rate}\n \
            num_train_epochs= {args.num_train_epochs}\n \
            per_device_train_batch_size= {args.train_batch_size} \n \
            per_device_eval_batch_size= {args.eval_batch_size}\n \
            logging_steps= {args.logging_steps}\n \
            output_dir= {args.output_dir}")

    if args.dataset_identifier == GHOSH_IDENTIFIER:

        ghosh_train_encoded, ghosh_dev_encoded, ghosh_test_encoded, id2label = get_ghosh_encoded_dataset()

        config = BertConfig.from_pretrained(
            "bert-base-uncased",
            id2label=id2label,
        )
        model = BertAdapterModel.from_pretrained(
            "bert-base-uncased",
            config=config,
        )

        if args.bert_model == CONT_BERT_PATH:
            print(f"Loading from pretrained Encoder from {args.bert_model}")
            contBert_saved_info = torch.load("./cont_pretrained_bert.ckpt_3", map_location=torch.device('cpu'))
            contBert_state_dict = contBert_saved_info["model_state_dict"]
            # contBert_state_dict['bert.embeddings.position_ids'] = model.state_dict()["bert.embeddings.position_ids"]
            model.load_state_dict(contBert_state_dict, strict=False)

        model.add_adapter(TASK2)
        model.add_classification_head(
            TASK2,
            num_labels = 2,
            id2label=id2label
        )
        model.train_adapter(TASK2)
        print(model.adapter_summary())


        training_args = TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            logging_steps=args.logging_steps,
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset= ghosh_train_encoded,
            eval_dataset= ghosh_dev_encoded,
            compute_metrics=compute_accuracy,
        )
        trainer.train()
        print("\n***** End of traing *****")
        print("***** Training dataset results *****")
        print(trainer.evaluate(ghosh_train_encoded))
        print("\n***** Eval dataset results *****")
        print(trainer.evaluate())
        print("\n***** Test dataset results *****")
        print(trainer.evaluate(ghosh_test_encoded))

        # Save the model and adapter
        model_dir = os.path.join(args.output_dir, "trained_model_after_adapter")
        try:
            os.mkdir(model_dir)
        except FileExistsError:
            os.mkdir(model_dir+"_copy")

        model_dir = os.path.join(args.output_dir, "trained_model_after_adapter", "model.ckpt")
        # Save model
        torch.save(model.state_dict(), model_dir)

        # Save adapter
        adapter_dir = os.path.join(args.output_dir, "adapter_ckpt")
        model.save_adapter(adapter_dir, TASK2)


        print("\n***** Test dataset results *****")
        print(trainer.evaluate(ghosh_test_encoded))

        print(f"learning_rate= {args.learning_rate}\n \
            num_train_epochs= {args.num_train_epochs}\n \
            per_device_train_batch_size= {args.train_batch_size} \n \
            per_device_eval_batch_size= {args.eval_batch_size}\n \
            logging_steps= {args.logging_steps}\n \
            output_dir= {args.output_dir}")
    



if __name__ == "__main__":
    args = getArgs()
    main(args)