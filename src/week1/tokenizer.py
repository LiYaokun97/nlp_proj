import torchtext
import torch
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import gensim

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


def getWord2VecModel(train_dataSet, test_dataSet, tokenizer):
    sentences = []
    keys = ["document_plaintext", "question_text"]
    for element in train_dataSet:
        for key in keys:
            sentences.append(tokenizer(element[key]))
    for element in test_dataSet:
        for key in keys:
            sentences.append(tokenizer(element[key]))
    w2v_model = gensim.models.Word2Vec(sentences, vector_size=128, min_count=1, window=3, epochs=15)

    w2v_model.wv.save_word2vec_format("vector.txt", binary=False)
    return w2v_model


def data_process(dataSet, w2vModel, tokenizer, tokenPart="document"):
    data = []
    for element in dataSet:
        if tokenPart == "document":
            en_tensor_ = torch.tensor(
                [w2vModel.get_vector(token) for token in tokenizer(element["document_plaintext"])], dtype=torch.float32)
            en_tensor_ = torch.mean(en_tensor_, dim=0, keepdim=True).cuda()
            data.append(en_tensor_)
        elif tokenPart == "question":
            en_tensor_ = torch.tensor([w2vModel.get_vector(token) for token in tokenizer(element["question_text"])],
                                      dtype=torch.float32)
            en_tensor_ = torch.mean(en_tensor_, dim=0, keepdim=True).cuda()
            data.append(en_tensor_)
        elif tokenPart == "answer":
            if (element["annotations"]["answer_start"] == [-1]):
                # en_tensor_ = torch.tensor([w2vModel.get_vector(token) for token in tokenizer(element["annotations"]["answer_text"])], dtype=torch.float32)
                # data.append(en_tensor_)
                data.append(torch.tensor([0], dtype=torch.int64).cuda())
            else:
                data.append(torch.tensor([1], dtype=torch.int64).cuda())
    return torch.cat(data, dim=0)


def getEnglishData(data, tokenPart="document"):
    tokenizer = get_tokenizer('basic_english', language="en")
    englishDataSet = getEnglishDataSet(data)
    englishVocab = build_vocab(englishDataSet, tokenizer)
    # w2vModel=getWord2VecModel(train_set,tokenizer)
    w2vModel = gensim.models.KeyedVectors.load_word2vec_format("vector.txt", binary=False)
    return data_process(englishDataSet, w2vModel, tokenizer, tokenPart)


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

