#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    : preprocess.py.py
@IDE     : PyCharm
@Author  : Yaokun Li
@Date    : 2022/10/20 15:48
@Description :
'''
import torch
import numpy as np
from torch.utils.data import Dataset
from bpemb import BPEmb
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")


dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

def getLanguageDataSet(data, language):
    def printAndL(x):
        return x["language"] == language
    return data.filter(printAndL)

def getEnglishDataSet(data):
    return getLanguageDataSet(data, "english")

# keep only english data
english_train_set = getEnglishDataSet(train_set)
english_validation_set = getEnglishDataSet(validation_set)

# delete useless data
english_train_set = english_train_set.remove_columns(["document_title", "language", "document_url"])
english_validation_set = english_validation_set.remove_columns(["document_title", "language", "document_url"])

def label_map_func(examples):
    labels = []
    for i in examples["annotations"]:
        if i["answer_start"] == [-1]:
            labels.append([0])

        else:
            labels.append([1])
    return {"label": labels}

english_label_train_set = english_train_set.map(label_map_func , batched=True, num_proc=4, remove_columns=["annotations"])
english_label_val_set = english_validation_set.map(label_map_func , batched=True, num_proc=4, remove_columns=["annotations"])

transformer_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

block_size = 128

def document_tokenize_function(examples):
    return transformer_tokenizer(examples['document_plaintext'], padding=True)

def question_tokenize_function(examples):
    return transformer_tokenizer(examples['question_text'], padding=True)

def switch_doc_name1(examples):
    return {"document_input_ids": examples['input_ids']}

def switch_doc_name2(examples):
    return {"document_attention_mask": examples['attention_mask']}

def switch_ques_name1(examples):
    return {"question_input_ids": examples['input_ids']}

def switch_ques_name2(examples):
    return {"question_attention_mask": examples['attention_mask']}

def get_final_dataset(data_set):
    return data_set\
        .map(document_tokenize_function, batched=True, num_proc=4, remove_columns=["document_plaintext"])\
        .map(switch_doc_name1, batched=True, num_proc=4, remove_columns=["input_ids"])\
        .map(switch_doc_name2, batched=True, num_proc=4, remove_columns=["attention_mask"])\
        .map(question_tokenize_function, batched=True, num_proc=4, remove_columns=["question_text"])\
        .map(switch_ques_name1, batched=True, num_proc=4, remove_columns=["input_ids"])\
        .map(switch_ques_name2, batched=True, num_proc=4, remove_columns=["attention_mask"])


train_dl = get_final_dataset(english_label_train_set).remove_columns("token_type_ids")
val_dl = get_final_dataset(english_label_val_set).remove_columns("token_type_ids")


def collate_batch_bilstm(dataset):
    label = []
    document = []
    document_mask = []
    question = []
    question_mask = []
    for data in dataset:
        label.append(data["label"])
        document.append(data["document_input_ids"])
        document_mask.append(data["document_attention_mask"])
        question.append(data["question_input_ids"])
        question_mask.append(data["question_attention_mask"])

    label = torch.Tensor(label).cuda()
    document = torch.Tensor(document).cuda()
    document_mask = torch.tensor(document_mask).cuda()
    question = torch.tensor(question).cuda()
    question_mask = torch.tensor(question_mask).cuda()

    return document, document_mask, question, question_mask, label


train_dl = torch.utils.data.DataLoader(train_dl, batch_size=32, collate_fn=collate_batch_bilstm)
valid_dl = torch.utils.data.DataLoader(val_dl, batch_size=32, collate_fn=collate_batch_bilstm)

for batch in train_dl:
    print(batch)