from lib2to3.pgen2 import token
from pickletools import optimize
import tokenizer
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from args import *
from model import *
import time
import nni
import warnings

warnings.filterwarnings('ignore')
params = vars(get_args)
tuner_params = nni.get_next_parameter()
params.update(tuner_params)
args = get_args()
dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]
validation_set = dataset["validation"]

english_answer_train_set = tokenizer.getEnglishData(train_set, tokenPart="answer")
print("finish load answer")
start_time = time.time()
english_question_train_set = tokenizer.getEnglishData(train_set, tokenPart="question")
end_time = time.time() - start_time
print("finish load question, cost time:", end_time)
start_time = time.time()
english_document_train_set = tokenizer.getEnglishData(train_set, tokenPart="document")
end_time = time.time() - start_time
print("finish load document, cost time:", end_time)

english_answer_validation_set = tokenizer.getEnglishData(validation_set, tokenPart="answer")
english_question_validation_set = tokenizer.getEnglishData(validation_set, tokenPart="question")
english_document_validation_set = tokenizer.getEnglishData(validation_set, tokenPart="document")

torch_dataset = Data.TensorDataset(english_question_train_set, english_document_train_set, english_answer_train_set)
train_loader = Data.DataLoader(dataset=torch_dataset,
                               batch_size=params['batch_size'],
                               shuffle=True)

criterion = nn.CrossEntropyLoss(reduction="sum")  # loss function

model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], amsgrad=True)

max_acc = 0
losses = []
for epoch in range(args.epochs):
    model.train()
    batch_num = 0
    for question_vec, document_vec, label in train_loader:
        predict_label = model(question_vec, document_vec)
        loss = criterion(predict_label, label)

        pred = predict_label.max(-1, keepdim=True)[1]
        acc = pred.eq(label.view_as(pred)).sum().item() / predict_label.shape[0]
        optimizer.zero_grad()
        if (acc > max_acc):
            max_acc = acc
            torch.save(model.state_dict(), 'model.pth')
        loss.backward()
        optimizer.step()
        batch_num += 1
        nni.report_intermediate_result(max_acc)
        print("epoch:", epoch + 1, "batch_num:", batch_num, "loss:", round(loss.item(), 4), "acc:", acc)
    losses.append(loss.item())

print("max_acc:", max_acc)
model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
model.load_state_dict(torch.load("model.pth"))
plt.plot(range(len(losses)), losses)
plt.savefig("loss.png")
predict_label = model(english_question_validation_set, english_document_validation_set)
pred = predict_label.max(-1, keepdim=True)[1]
label = english_answer_validation_set
test_acc = pred.eq(label.view_as(pred)).sum().item() / predict_label.shape[0]
nni.report_final_result(test_acc)
print("test acc:", test_acc)

# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

# todo 建立特征 根据特征预测问题是否可回答