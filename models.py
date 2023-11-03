import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertModel


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert_cross = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)

    def forward(self, text_ids, text_mask, know_ids, know_mask, labels=None):
        # text_info, pooled_text_info = self.bert_cross(input_ids=text_ids, attention_mask=text_mask)
        text_info, pooled_text_info = self.bert_cross(input_ids=know_ids, attention_mask=know_mask)
        res = self.dropout(pooled_text_info)
        logits = self.classifier(res)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class Bert_concat(nn.Module):
    def __init__(self):
        super(Bert_concat, self).__init__()
        self.text_bert = BertModel.from_pretrained('bert-base-uncased')
        self.know_bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768 * 2, 2)

    def forward(self, text_ids, text_mask, know_ids, know_mask, labels=None):
        text_info, pooled_text_info = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        know_info, pooled_know_info = self.know_bert(input_ids=know_ids, attention_mask=know_mask)
        res = torch.cat([pooled_text_info, pooled_know_info], dim=1)
        res = self.dropout(res)
        logits = self.classifier(res)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertOutput(nn.Module):
    def __init__(self):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(3072, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    def __init__(self):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(768, 3072)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertSelfOutput(nn.Module):
    def __init__(self):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(768, 768)
        self.LayerNorm = BertLayerNorm(768, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class KLBert(nn.Module):
    def __init__(self):
        super(KLBert, self).__init__()
        self.text_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        self.know_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        self.W_gate = nn.Linear(768 * 2, 1)
        self.intermediate = BertIntermediate()
        self.output = BertSelfOutput()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.secode_output = BertOutput()

    def forward(self, text_ids, text_mask, know_ids, know_mask, labels=None):
        text_info, pooled_text_info = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        know_info, pooled_know_info = self.know_bert(input_ids=know_ids, attention_mask=know_mask)
        
        #print(know_info)
        # 32*40*768
        attn = torch.matmul(text_info, know_info.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        know_text = torch.matmul(attn, know_info)

        combine_info = torch.cat([text_info, torch.mean(know_info, dim=1).unsqueeze(1).expand(text_info.size(0),
                                                                                              text_info.size(1),
                                                                                              text_info.size(-1))],
                                 dim=-1)
        alpha = self.W_gate(combine_info)
        alpha = F.sigmoid(alpha)

        # 32*1*768
        text_info = torch.matmul(alpha.transpose(1, 2), text_info)
        # 32*1*768
        know_text = torch.matmul((1 - alpha).transpose(1, 2), know_text)
        # 32*1*768

        #no gate
        ################################################
        # res = torch.cat([text_info.squeeze(1), know_text.squeeze(1)],dim=1)
        ################################################


        res = self.output(know_text, text_info)

        # 32*1*3072
        # no-gate
        ################################################
        # res = torch.mean(res,dim=1)
        ################################################
        intermediate_res = self.intermediate(res)
        # 32*1*768
        res = self.secode_output(intermediate_res, res)


        # 32*40*768
        # res = text_info+know_text
        # res = torch.mean(res,dim=1)
        logits = self.classifier(res)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits

class ContBert(nn.Module):
    def __init__(self):
        super(ContBert, self).__init__()
        self.text_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        saved_info = torch.load("./cont_pretrained_bert.ckpt_3", map_location=torch.device('cpu'))

        state_dict = saved_info["model_state_dict"]
        state_dict = {key.replace("bert.", ""):val for key, val in state_dict.items() if "cls" not in key}
        self.text_bert.load_state_dict(state_dict)
        
        self.know_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        self.know_bert.load_state_dict(state_dict)

        self.W_gate = nn.Linear(768 * 2, 1)
        self.intermediate = BertIntermediate()
        self.output = BertSelfOutput()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.secode_output = BertOutput()

    def forward(self, text_ids, text_mask, know_ids, know_mask, labels=None):
        text_info, pooled_text_info = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        know_info, pooled_know_info = self.know_bert(input_ids=know_ids, attention_mask=know_mask)

        #print(know_info)
        # 32*40*768
        attn = torch.matmul(text_info, know_info.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        know_text = torch.matmul(attn, know_info)

        combine_info = torch.cat([text_info, torch.mean(know_info, dim=1).unsqueeze(1).expand(text_info.size(0),
                                                                                              text_info.size(1),
                                                                                              text_info.size(-1))],
                                 dim=-1)
        alpha = self.W_gate(combine_info)
        alpha = F.sigmoid(alpha)

        # 32*1*768
        text_info = torch.matmul(alpha.transpose(1, 2), text_info)
        # 32*1*768
        know_text = torch.matmul((1 - alpha).transpose(1, 2), know_text)
        # 32*1*768

        #no gate
        ################################################
        # res = torch.cat([text_info.squeeze(1), know_text.squeeze(1)],dim=1)
        ################################################


        res = self.output(know_text, text_info)

        # 32*1*3072
        # no-gate
        ################################################
        # res = torch.mean(res,dim=1)
        ################################################
        intermediate_res = self.intermediate(res)
        # 32*1*768
        res = self.secode_output(intermediate_res, res)


        # 32*40*768
        # res = text_info+know_text
        # res = torch.mean(res,dim=1)
        logits = self.classifier(res)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits


class OurBert(nn.Module):
    def __init__(self):
        super(OurBert, self).__init__()
        # Jane: Added to account for contraction and entailment
        self.flute_bert = torch.load("./flute_here/flute_checkpoint.pt", map_location=torch.device('cpu')).bert
        self.W_proj = nn.Linear(768, 40)

        self.text_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        self.know_bert = BertModel.from_pretrained('bert-base-uncased', return_dict = False)
        self.W_gate = nn.Linear(768 * 2, 1)
        self.intermediate = BertIntermediate()
        self.output = BertSelfOutput()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.secode_output = BertOutput()

    def forward(self, text_ids, text_mask, flute_ids, flute_mask,know_ids, know_mask, labels=None):
        text_info, pooled_text_info = self.text_bert(input_ids=text_ids, attention_mask=text_mask)
        
        # Use learnable W_proj to project flute_bert info to text_info
        flute_info, pooled_text_info = self.flute_bert(input_ids = flute_ids, attention_mask = flute_mask).values()
        projection = self.W_proj(flute_info)

        #print(f"projection: {projection.shape}")
        #print(f"text_info: {text_info.shape}")
        text_info = torch.matmul(projection, text_info)

        know_info, pooled_know_info = self.know_bert(input_ids=know_ids, attention_mask=know_mask)

        #print(know_info)
        # 32*40*768
        attn = torch.matmul(text_info, know_info.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        know_text = torch.matmul(attn, know_info)

        combine_info = torch.cat([text_info, torch.mean(know_info, dim=1).unsqueeze(1).expand(text_info.size(0),
                                                                                              text_info.size(1),
                                                                                              text_info.size(-1))],
                                 dim=-1)
        alpha = self.W_gate(combine_info)
        alpha = F.sigmoid(alpha)

        # 32*1*768
        text_info = torch.matmul(alpha.transpose(1, 2), text_info)
        # 32*1*768
        know_text = torch.matmul((1 - alpha).transpose(1, 2), know_text)
        # 32*1*768

        #no gate
        ################################################
        # res = torch.cat([text_info.squeeze(1), know_text.squeeze(1)],dim=1)
        ################################################


        res = self.output(know_text, text_info)

        # 32*1*3072
        # no-gate
        ################################################
        # res = torch.mean(res,dim=1)
        ################################################
        intermediate_res = self.intermediate(res)
        # 32*1*768
        res = self.secode_output(intermediate_res, res)


        # 32*40*768
        # res = text_info+know_text
        # res = torch.mean(res,dim=1)
        logits = self.classifier(res)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits
