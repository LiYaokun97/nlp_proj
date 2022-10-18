import torch
from args import *
import numpy as np
from torch.utils.data import Dataset
from bpemb import BPEmb
# import nni
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from datasets import load_dataset
import transformers
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer
from model import *
from transformers import GPT2Tokenizer, GPT2Model

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args = get_args()

# squad_v2等于True或者False分别代表使用SQUAD v1 或者 SQUAD v2。
# 如果您使用的是其他数据集，那么True代表的是：模型可以回答“不可回答”问题，也就是部分问题不给出答案，而False则代表所有问题必须回答。
squad_v2 = False

batch_size = 16

dataset = load_dataset("copenlu/answerable_tydiqa")

from transformers import AutoTokenizer

tokenizer = GPT2Tokenizer.from_pretrained('Finnish-NLP/gpt2-finnish')


def getLanguageDataSet(data, language):
    def printAndL(x):
        return x["language"] == language

    return data.filter(printAndL)


def getEnglishDataSet(data):
    return getLanguageDataSet(data, "english")


def getJapDataSet(data):
    return getLanguageDataSet(data, "japanese")


def getFinDataSet(data):
    return getLanguageDataSet(data, "finnish")


finnish_set = getFinDataSet(dataset)
pad_on_right = True
max_length = 128
doc_stride = 64

finnish_set = finnish_set.remove_columns("language")

finnish_set = finnish_set.remove_columns("document_url")


def prepare_train_features(examples):
    # 既要对examples进行truncation（截断）和padding（补全）还要还要保留所有信息，所以要用的切片的方法。
    # 每一个一个超长文本example会被切片成多个输入，相邻两个输入之间会有交集。
    tokenized_examples = tokenizer(
        examples["question_text" if pad_on_right else "document_plaintext"],
        examples["document_plaintext" if pad_on_right else "question_text"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        padding="max_length",
    )

    if examples["annotations"]["answer_start"] == [-1]:
        tokenized_examples["labels"] = [0]
    else:
        tokenized_examples["labels"] = [1]
    return tokenized_examples


def get_torch_vec(examples):
    train_data = {}
    train_data["input_ids"] = torch.tensor(examples["input_ids"]).cuda()
    train_data["attention_mask"] = torch.tensor(examples["attention_mask"]).cuda()
    train_data["labels"] = torch.tensor(examples["labels"]).cuda()
    return train_data


tokenized_datasets = finnish_set.map(prepare_train_features, batched=False,
                                     remove_columns=finnish_set["train"].column_names)
tokenized_datasets = tokenized_datasets.map(get_torch_vec)

train_data = tokenized_datasets['train']
test_data = tokenized_datasets['validation']

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True)

test_loader = Data.DataLoader(dataset=test_data,
                              batch_size=16,
                              shuffle=True)

criterion = torch.nn.CrossEntropyLoss(reduction="mean")  # loss function

model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

max_acc = 0
losses = []

for epoch in range(args.epochs):
    model.train()
    batch_num = 0

    for batch in train_loader:
        label = torch.stack(batch['labels'], dim=1).cuda()
        label = torch.squeeze(label, dim=-1).cuda()
        predict_label = model(batch)
        loss = criterion(predict_label, label)

        pred = predict_label.max(-1, keepdim=True)[1]
        acc = pred.eq(label.view_as(pred)).sum().item() / predict_label.shape[0]
        optimizer.zero_grad()
        if (acc > max_acc):
            max_acc = acc
            torch.save(model.state_dict(), 'model_week3_fin.pth')
        loss.backward()
        optimizer.step()
        batch_num += 1
        # nni.report_intermediate_result(max_acc)
        print("epoch:", epoch + 1, "batch_num:", batch_num, "loss:", round(loss.item(), 4), "acc:", acc)
        losses.append(loss.item())

plt.plot([i for i in range(len(losses))], losses)
plt.savefig("loss_week3.png")
print("train_max_acc:", max_acc)
model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
model.load_state_dict(torch.load("model_week3_fin.pth"))

count_acc = 0
max_test_acc = 0
count = 0
real_labels = 0
predict_labels = 0
for batch in test_loader:
    label = torch.stack(batch['labels'], dim=0).cuda().squeeze(0)
    predict_label = model(batch)
    if (count == 0):
        real_labels = label.cpu().detach().numpy()
        predict_labels = predict_label.max(-1, keepdim=True)[1].squeeze(1).cpu().detach().numpy()
    else:
        real_labels = np.hstack((real_labels, label.cpu().detach().numpy()))
        predict_labels = np.hstack(
            (predict_labels, predict_label.max(-1, keepdim=True)[1].squeeze(1).cpu().detach().numpy()))
    count += 1
    # pred = predict_label.max(-1,keepdim=True)[1]
    # test_acc = pred.eq(label.view_as(pred)).sum().item()/predict_label.shape[0]
    # count+=1
    # max_test_acc = max(test_acc,max_test_acc)
    # count_acc+= test_acc

# real = np.concatenate(real_labels,axis=1)
# predict_labels = np.array(predict_labels)
report = classification_report(predict_labels, real_labels, output_dict=True)
print(pd.DataFrame(report).transpose())


