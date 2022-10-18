from args import *
from transformers import AutoTokenizer
from model import *
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from src.week5 import data_processing

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args = get_args()

import torch
from torch import nn
from datasets import load_dataset, load_metric

squad_v2 = False
# distilbert-base-uncased can only be used in English
model_checkpoint = "distilbert-base-uncased"
# model_checkpoint = "bert-base-multilingual-uncased"
batch_size = 16

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset = load_dataset("copenlu/answerable_tydiqa")
train_set = dataset["train"]


train_loader = data_processing.get_train_loader()
test_loader = data_processing.get_test_loader()

criterion = torch.nn.CrossEntropyLoss(reduction="mean")  # loss function

model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

max_acc = 0
losses = []


def predict_same_with_label(predict, label):
    if label.equal(predict):
        return 1
    else:
        return 0


for epoch in range(args.epochs):
    model.train()
    batch_num = 0

    for batch in train_loader:
        label = torch.stack(batch['label'], dim=1).cuda()
        label = torch.reshape(label, (-1, 1)).squeeze(-1)
        predict_label = model(batch)
        loss = criterion(predict_label, label)

        pred = predict_label.max(-1, keepdim=True)[1]
        acc = predict_same_with_label(pred, label)
        optimizer.zero_grad()
        if (acc > max_acc):
            max_acc = acc
            torch.save(model.state_dict(), 'model.pth')
        loss.backward()
        optimizer.step()
        batch_num += 1
        # nni.report_intermediate_result(max_acc)
        print("epoch:", epoch + 1, "batch_num:", batch_num, "loss:", round(loss.item(), 4), "acc:", acc)
    losses.append(loss.item())

print("train_max_acc:", max_acc)
model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
model.load_state_dict(torch.load("model.pth"))


def evaluate(model: nn.Module, valid_dl):
    """
    Evaluates the model on the given dataset
    :param model: The model under evaluation
    :param valid_dl: A `DataLoader` reading validation data
    :return: The accuracy of the model on the dataset
    """
    # VERY IMPORTANT: Put your model in "eval" mode -- this disables things like
    # layer normalization and dropout
    model.eval()
    labels_all = []
    preds_all = []

    # ALSO IMPORTANT: Don't accumulate gradients during this process
    with torch.no_grad():
        for batch in valid_dl:
            # batch = tuple(t.to(device) for t in batch)
            labels = torch.stack(batch['label'], dim=1).cuda()
            hidden_states = None

            logits = model(batch)
            preds_all.extend(torch.argmax(logits, dim=-1).reshape(-1).detach().cpu().numpy())
            labels_all.extend(labels.reshape(-1).detach().cpu().numpy())

    P, R, F1, _ = precision_recall_fscore_support(labels_all, preds_all, average='macro')
    print(confusion_matrix(labels_all, preds_all))
    return F1


F1 = evaluate(model, test_loader)
print("F1:", F1)


def transformer_sampling(model, sentence, max_len):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=max_len, do_sample=True)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))
