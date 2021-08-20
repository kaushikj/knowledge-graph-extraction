#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import os
import sys
import json
import numpy as np
from transformers import BertModel, BertTokenizer
from nltk.tokenize import word_tokenize


# # Build BERT Encoder
# Create a BERT encoder class which loads a pretrained Model. It tokenizes the text and converts it to a numerical encoding, this encoding along with attention mask is fed to the Softmax neural network.

# In[2]:


class BERTEncoder(torch.nn.Module):
    def __init__(self, max_length, pretrain_path):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.max_length = max_length
        self.blank_padding = True
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)

    def forward(self, token, att_mask):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
        Return:
            (B, H), representations for sentences
        """
        _, x = self.bert(token, attention_mask=att_mask)
        return x

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'token' not in item:
            raise Exception('Check your input parameters')

        sentence = item['token']
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        pos_min = pos_head
        pos_max = pos_tail
        if pos_head[0] > pos_tail[0]:
            pos_min = pos_tail
            pos_max = pos_head
            rev = True
        else:
            rev = False
        
        sent_left = self.tokenizer.tokenize(' '.join(sentence[:pos_min[0]]))
        ent0 = self.tokenizer.tokenize(' '.join(sentence[pos_min[0]:pos_min[1]]))
        sent_middle = self.tokenizer.tokenize(' '.join(sentence[pos_min[1]:pos_max[0]]))
        ent1 = self.tokenizer.tokenize(' '.join(sentence[pos_max[0]:pos_max[1]]))
        sent_right = self.tokenizer.tokenize(' '.join(sentence[pos_max[1]:]))

        ent0 = ['[unused0]'] + ent0 + ['[unused1]'] if not rev else ['[unused2]'] + ent0 + ['[unused3]']
        ent1 = ['[unused2]'] + ent1 + ['[unused3]'] if not rev else ['[unused0]'] + ent1 + ['[unused1]']

        re_tokens = ['[CLS]'] + sent_left + ent0 + sent_middle + ent1 + sent_right + ['[SEP]']
#         print('re_tokens', re_tokens)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1
        return indexed_tokens, att_mask


# # Build Softmax Neural Network

# In[3]:


class SoftmaxNN(torch.nn.Module):
    def __init__(self, sentence_encoder, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = len(rel2id)
        self.fc = torch.nn.Linear(self.sentence_encoder.hidden_size,self.num_class)
        self.softmax = torch.nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = torch.nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, item):
        self.eval()
        _item = self.sentence_encoder.tokenize(item)
        item = []
        for x in _item:
            item.append(x.to(next(self.parameters()).device))
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score
    
    def forward(self, *args):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(*args) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep) # (B, N)
        return logits


# In[4]:


def get_model_bert_wiki80():
    model_name = 'wiki80_bert_softmax'
    rel2id = json.load(open('./benchmark/wiki80/wiki80_rel2id.json'))
    sentence_encoder = BERTEncoder(
        max_length=80, pretrain_path='./pretrain/bert-base-uncased')
    model = SoftmaxNN(sentence_encoder, rel2id)
    ckpt = './pretrain/nre/wiki80_bert_softmax.pth.tar'
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'])        
    return model


# # Test The model

# In[5]:


model = get_model_bert_wiki80()


# In[6]:


text = 'He was the son of Mael Duin mac Maele Fithrich, and grandson of the high king aed Uaridnach (died 612).'
tokens = word_tokenize(text)
for i in range(len(tokens)):
    print(i, tokens[i])


# In[7]:


result = model.infer({'token': tokens, 'h': {'pos': (5, 10)}, 't': {'pos': (17,19)}})
print(result)

#Check if model is correctly working
if result[0] != 'father':
    raise Exception('Not father')
if result[1] < 0.69:
    raise Exception('Accuracy drop')


# In[ ]:




