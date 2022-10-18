import torch.utils.data as Data
from transformers import AutoTokenizer

import torch
from torch import nn
from datasets import load_dataset, load_metric

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dataset = load_dataset("copenlu/answerable_tydiqa")
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
import transformers
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

def getLanguageDataSet(data, language):
    def printAndL(x):
        return x["language"] == language
    return data.filter(printAndL)


def getEnglishDataSet(data):
    return getLanguageDataSet(data, "english")

# keep only english data
english_set = getEnglishDataSet(dataset)
english_set = english_set.remove_columns("language")
english_set = english_set.remove_columns("document_url")


max_length = 256  # 输入feature的最大长度，question和context拼接之后
doc_stride = 64  # 2个切片之间的重合token数量。
pad_on_right = True


def get_answer_index(start_index, offset_mapping):
    res = -1
    for i, t in enumerate(offset_mapping):
        if t[0] <= start_index and t[1] >= start_index:
            res = i
    return res


def get_start_and_end_answer_index(start_index, end_index, offset_mapping):
    start_res = -1
    end_res = -1
    for i, t in enumerate(offset_mapping):
        if t[0] <= start_index and t[1] >= start_index:
            start_res = i
        if t[0] <= end_index and t[1] >= end_index:
            end_res = i
    return (start_res, end_res)


def prepare_train_features(examples):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question_text" if pad_on_right else "document_plaintext"],
        examples["document_plaintext" if pad_on_right else "question_text"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 我们使用overflow_to_sample_mapping参数来映射切片片ID到原始ID。
    # 比如有2个expamples被切成4片，那么对应是[0, 0, 1, 1]，前两片对应原来的第一个example。
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # offset_mapping也对应4片
    # offset_mapping参数帮助我们映射到原始输入，由于答案标注在原始输入上，所以有助于我们找到答案的起始和结束位置。
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # 重新标注数据
    tokenized_examples["labels"] = []

    for i, offsets in enumerate(offset_mapping):
        # 对每一片进行处理
        # 将无答案的样本标注到CLS上
        input_ids = tokenized_examples["input_ids"][i]

        # 区分question和context
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 拿到原始的example 下标.
        sample_index = sample_mapping[i]
        answers = examples["annotations"][sample_index]
        # 如果没有答案，则使用CLS所在的位置为答案.
        if len(answers["answer_start"]) == [-1]:
            tokenized_examples["labels"].append([0] * max_length)
        else:
            # 答案的character级别Start/end位置.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["answer_text"][0])

            # 找到token级别的index start.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # 找到token级别的index end.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1
            # 检测答案是否超出文本长度，超出的话也适用CLS index作为标注.
            temp = [0] * max_length
            if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                tokenized_examples["labels"].append(temp)
            elif offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                s, e = get_start_and_end_answer_index(start_char, end_char, offsets)
                temp[s] = 1
                temp[s + 1:e + 1] = [2] * (e - s)
                tokenized_examples["labels"].append(temp)
            elif offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] < end_char:
                s = get_answer_index(start_char, offsets)
                temp[s] = 1
                length = len(temp) - s
                temp[s:] = [2] * length
                tokenized_examples["labels"].append(temp)
            elif offsets[token_start_index][0] > start_char and offsets[token_end_index][1] >= end_char:
                e = get_answer_index(end_char, offsets)
                temp[:e + 1] = [2] * e
                tokenized_examples["labels"].append(temp)
    return tokenized_examples


tokenized_datasets = english_set.map(prepare_train_features, batched=True,
                                     remove_columns=english_set["train"].column_names)

# tokenized_datasets = tokenized_datasets.remove_columns("token_type_ids")

train_data = tokenized_datasets['train']
test_data = tokenized_datasets['validation']

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=32,
                               shuffle=True)

test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=32,
                              shuffle=True)



