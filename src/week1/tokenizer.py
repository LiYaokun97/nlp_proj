import torchtext
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

def getLanguageDataSet(data, language):
    return data.filter(lambda x: x['language'] == language)

def getJapaneseDataSet(data):
    return getLanguageDataSet(data, "japanese")

def getEnglishDataSet(data):
    return getLanguageDataSet(data, "english")

def getFinnishDataSet(data):
    return getLanguageDataSet(data, "finnish")

def build_vocab(dataSet, tokenizer):
  counter = Counter()
  for data in dataSet:
    counter.update(tokenizer(data['document_plaintext']))
  return Vocab(counter)

def data_process(dataSet, vocab, tokenizer, tokenPart = "document"):
  data = []
  for element in dataSet:
    if tokenPart == "document":
        en_tensor_ = torch.tensor([vocab[token] for token in tokenizer(element["document_plaintext"])], dtype=torch.long)
        data.append(en_tensor_)
    elif tokenPart == "question":
        en_tensor_ = torch.tensor([vocab[token] for token in tokenizer(element["question_text"])],dtype=torch.long)
        data.append(en_tensor_)
    elif tokenPart == "answer":
        if len(element["annotations"]["answer_text"]) != 0:
            en_tensor_ = torch.tensor([vocab[token] for token in tokenizer(element["annotations"]["answer_text"])], dtype=torch.long)
            data.append(en_tensor_)
        else:
            data.append(torch.tensor([0], dtype=torch.long))
  return data

def getEnglishData(data, tokenPart = "document"):
    tokenizer = get_tokenizer('basic_english', language="en")
    englishDataSet = getEnglishDataSet(data)
    englishVocab = build_vocab(englishDataSet, tokenizer)
    return data_process(englishDataSet, englishVocab, tokenizer, tokenPart)

# todo Japanese的tokenizer
def getJapaneseData(data):
    tokenizer = get_tokenizer('basic_english', language="en")
    dataset = getJapaneseDataSet(data)
    vocab = build_vocab(dataset, tokenizer)
    return data_process(dataset, vocab, tokenizer)

# todo Finnish的tokenizer
def getFinnishData(data):
    tokenizer = get_tokenizer('basic_english', language="en")
    dataset = getFinnishDataSet(data)
    vocab = build_vocab(dataset, tokenizer)
    return data_process(dataset, vocab, tokenizer)

