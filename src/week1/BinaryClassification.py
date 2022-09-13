import tokenizer
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

english_document_train_set = tokenizer.getEnglishData(train_set, tokenPart="document")
english_answer_train_set = tokenizer.getEnglishData(train_set, tokenPart="answer")
# todo 建立特征 根据特征预测问题是否可回答