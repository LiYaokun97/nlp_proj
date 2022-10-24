import torch
from args import *
import numpy as np
from torch.utils.data import Dataset
from bpemb import BPEmb
# import nni
import tqdm
import torch.utils.data as Data
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer
from model import *
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args = get_args()

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

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
doc_stride = 128  # 2个切片之间的重合token数量。
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
    tokenized_examples["label"] = []

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
            tokenized_examples["label"].append([0] * max_length)
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
                tokenized_examples["label"].append(temp)
            elif offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char:
                s, e = get_start_and_end_answer_index(start_char, end_char, offsets)
                temp[s] = 1
                temp[s + 1:e + 1] = [2] * (e - s)
                tokenized_examples["label"].append(temp)
            elif offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] < end_char:
                s = get_answer_index(start_char, offsets)
                temp[s] = 1
                length = len(temp) - s
                temp[s:] = [2] * length
                tokenized_examples["label"].append(temp)
            elif offsets[token_start_index][0] > start_char and offsets[token_end_index][1] >= end_char:
                e = get_answer_index(end_char, offsets)
                temp[:e + 1] = [2] * e
                tokenized_examples["label"].append(temp)
    return tokenized_examples


def predict_same_with_label(predict, label):
    if label.equal(predict):
        return 1
    else:
        return 0


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

criterion = torch.nn.CrossEntropyLoss(reduction="mean")  # loss function

model = QA_model(args.input_dim, args.hidden_dim, args.output_dim).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

max_acc = 0
losses = []
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


# input_ids= [torch.tensor(ids).cuda() for ids in test_data['input_ids']]
# attenion_mask= [torch.tensor(mask).cuda() for mask in test_data['attention_mask']]
# test_data ={"input_ids":input_ids,"attention_mask":attenion_mask}
# predict_label = model(test_data)
# pred = predict_label.max(-1,keepdim=True)[1]
# label = torch.squeeze(test_data['label'],dim=-1).cuda()
# test_acc = pred.eq(label.view_as(pred)).sum().item()/predict_label.shape[0]
# print("test acc:",test_acc)
# max_acc = 0
# count=0
# count_acc=0


# params = vars(get_args)
# tuner_params = nni.get_next_parameter()
# params.update(tuner_params)


# # model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# # unsupervised_imdb = load_dataset('imdb', split='unsupervised')
# # unsupervised_imdb_splits = unsupervised_imdb.train_test_split(test_size=0.01)
# # block_size = 128
# # def tokenize_function(examples):
# #     return tokenizer(examples["text"])

# # def group_texts(examples):
# #     # Concatenate all texts.
# #     keys = ['attention_mask', 'input_ids']
# #     concatenated_examples = {k: sum(examples[k], []) for k in keys}
# #     total_length = len(concatenated_examples[list(keys)[0]])
# #     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
# #         # customize this part to your needs.
# #     total_length = (total_length // block_size) * block_size
# #     # Split by chunks of max_len.
# #     result = {
# #         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
# #         for k, t in concatenated_examples.items()
# #     }
# #     # this is needed as the used dataset is a subclass of ClassificationDataset, which requires label as a field...
# #     result["label"] = result["input_ids"].copy()
# #     result["labels"] = result["input_ids"].copy()
# #     return result
# # unsupervised_imdb_tok = unsupervised_imdb_splits.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
# # unsupervised_imdb_splits = unsupervised_imdb_splits.remove_columns(['label'])
# # unsupervised_imdb_tok_lm = unsupervised_imdb_tok.map(group_texts, batched=True, batch_size=1000, num_proc=4,)
# # unsupervised_imdb_tok_lm = unsupervised_imdb_tok_lm.remove_columns(['label'])

# # train_data = unsupervised_imdb_tok_lm['train'].remove_columns(["labels"])
# # train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
# # for batch in train_loader:

# #     batch['input_ids'] = torch.stack(batch['input_ids'],dim=0)
# #     batch['attention_mask'] = torch.stack(batch['attention_mask'],dim=0)
# #     model_out = model(**batch, output_hidden_states=True, return_dict=True)

def transformer_sampling(model, sentence, max_len):
    # encode context the generation is conditioned on
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    # generate text until the output length (which includes the context length) reaches 50
    greedy_output = model.generate(input_ids, max_length=max_len, do_sample=True)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# # def mean_pooling(model_output, attention_mask):
# #     # Mean Pooling - Take attention mask into account for correct averaging
# #     input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
# #     sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
# #     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
# #     return sum_embeddings / sum_mask


# # encoded = tokenizer(['Bromwell High is a cartoon comedy.'],
# #                     return_tensors='pt')
# # encoded = {k:v.to('cpu') for k, v in encoded.items()}

# # model_output = model(**encoded, output_hidden_states=True, return_dict=True)
# # print(model_output.keys())
# # print(len(model_output['hidden_states'])) # contextual representations of separate words from each of the 6 layes
# # print(model_output['hidden_states'][-1].shape) # last layer with contextual representations (batch_size x num words x representation dim)

# # # Aggregate all the representations into one
# # mean_model_output = mean_pooling(model_output['hidden_states'][-1], encoded['attention_mask'])


# dataset = load_dataset("copenlu/answerable_tydiqa")
# train_set = dataset["train"]
# validation_set = dataset["validation"]

# english_document_train_set = tokenizer.getEnglishData(train_set, tokenPart="document")

# english_answer_train_set = tokenizer.getEnglishData(train_set, tokenPart="answer")

# english_question_train_set = tokenizer.getEnglishData(train_set,tokenPart="question")


# # english_answer_validation_set = tokenizer.getEnglishData(validation_set, tokenPart="answer")
# # english_question_validation_set = tokenizer.getEnglishData(validation_set,tokenPart="question")
# # english_document_validation_set = tokenizer.getEnglishData(validation_set, tokenPart="document")

# torch_dataset = Data.TensorDataset(english_question_train_set,english_document_train_set,english_answer_train_set)
# train_loader = Data.DataLoader(dataset=torch_dataset,
#                                 batch_size=params['batch_size'],
#                                 shuffle=True)


# # print("max_acc:",max_acc)
# # model = QA_model(args.input_dim,args.hidden_dim,args.output_dim).to('cuda')
# # model.load_state_dict(torch.load("model.pth"))
# # plt.plot(range(len(losses)),losses)
# # plt.savefig("loss.png")
# # predict_label = model(english_question_validation_set,english_document_validation_set)
# # pred = predict_label.max(-1,keepdim=True)[1]
# # label = english_answer_validation_set
# # test_acc = pred.eq(label.view_as(pred)).sum().item()/predict_label.shape[0]
# # nni.report_final_result(test_acc)
# # print("test acc:",test_acc)


# # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)

# # todo 建立特征 根据特征预测问题是否可回答