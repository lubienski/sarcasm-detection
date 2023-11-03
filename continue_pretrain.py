import pandas as pd
import os
import numpy as np
from scipy.special import softmax
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

import torch
from sklearn.metrics import precision_recall_fscore_support


from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel,BertForPreTraining, BertTokenizer
from pretrain_utils import *

def split_text(text):
    sentences = [
        sentence for sentence in text.split(',') if sentence != ''
    ]
    sentences_period = [
        sentence for sentence in text.split('.') if sentence != ''
    ]
    num_sentences = len(sentences)
    num_sentences_perid = len(sentences_period)
    if num_sentences > 1:
        if num_sentences // 2 > 1:
            cutoff = num_sentences // 2
        else:
            cutoff = 1
        sentence_a = ",".join(sentences[:cutoff])
        sentence_b = ",".join(sentences[cutoff:])
        return (sentence_a, sentence_b, sentences)

    elif num_sentences_perid > 1:
        if num_sentences_perid // 2 > 1:
            cutoff = num_sentences_perid // 2
        else:
            cutoff = 1
        sentence_a = ".".join(sentences_period[:cutoff])
        sentence_b = ".".join(sentences_period[cutoff:])
        return (sentence_a, sentence_b, sentences_period)
    return -1

def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForPreTraining.from_pretrained("bert-base-uncased")
    # Load the dataset for Ghosh
    data_dir = "./data/Ghosh"
    pretrain_processer = ProcesserPretrain(data_dir = data_dir, max_seq_length = 512)
    train_examples = pretrain_processer.get_train_examples()
    generator = torch.Generator().manual_seed(42)
    a,b = torch.utils.data.random_split(train_examples, [0.9,0.1], generator = generator) 
    
    save_for_later = [train_examples[i] for i in a.indices]
    torch.save(save_for_later, "./90%_train_data_3.pt")
    pretrain_data = [train_examples[i] for i in b.indices]
    print(f"{len(pretrain_data)} of {len(train_examples)} train is drawn for continue pretrain")

    bag_of_text = [item.text for item in pretrain_data]
    bag_size = len(bag_of_text)

    # Prepare for NSP (next sentence prediction
    sentence_a = []
    sentence_b = []
    label = []
    

    for text in bag_of_text:
        result = split_text(text)

        if result != -1:
            sentence_1, sentence_2, sentences = result
            num_sentences = len(sentences)
            start = random.randint(0, num_sentences-2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentence_1)
                sentence_b.append(sentence_2)
                label.append(1)
            else:
                index = random.randint(0, bag_size-1)
                # this is NotNextSentence
                sentence_a.append(sentence_1)
                sentence_b.append(bag_of_text[index])
                label.append(0)

    inputs = tokenizer(sentence_a, sentence_b, \
            return_tensors="pt", max_length=512, \
            truncation=True, padding="max_length")
    inputs['next_sentence_label'] = torch.LongTensor([label]).T
    

    # MLM preparation
    inputs['labels'] = inputs.input_ids.detach().clone()
    inputs.keys()
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    # CLS(101) SEP(102), PAD(0)
    #  We don't want to mask these tokens
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
           (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    pretrain_dataset = PretrainDataset(inputs)
    loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size = 16, shuffle=True)
    
    device = torch.device("cuda:1")
    epochs = 8
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    model = model.to(device)
    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optimizer.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            next_sentence_label = batch['next_sentence_label'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        next_sentence_label=next_sentence_label,
                        labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optimizer.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        print(f"Epoch {epoch}: loss {loss}")
    torch.save(model, "cont_pretrained_bert_3.ckpt")    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, \
            "./cont_pretrained_bert.ckpt_3")
if __name__ == "__main__":
    main()

