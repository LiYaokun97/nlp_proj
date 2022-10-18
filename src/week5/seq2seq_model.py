#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    : seq2seq_model.py
@IDE     : PyCharm 
@Author  : Yaokun Li
@Date    : 2022/10/27 11:37 
@Description : 
'''


import torch.utils.data as Data
from transformers import AutoTokenizer

import torch
from torch import nn
from datasets import load_dataset, load_metric
import numpy as np
from torch.utils.data import Dataset, DataLoader
import io

from torch.optim import Adam
import io
from math import log
from numpy import array
from numpy import argmax
import torch
import random
from math import log
from numpy import array
from numpy import argmax
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torchcrf import CRF
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from typing import List, Tuple, AnyStr
from tqdm import tqdm_notebook as tqdm
from tqdm import notebook
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import load_dataset, load_metric
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import heapq


dataset = load_dataset("copenlu/answerable_tydiqa")


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

from torchtext.data import get_tokenizer
torch_tokenizer = get_tokenizer('basic_english', language="en")

def tokenizer_data(examples):
    answer_start = examples["annotations"]["answer_start"][0]
    if answer_start == -1:
        tokens_question = torch_tokenizer(examples["question_text"]) + ["[SEP]"]
        tokens_doc = torch_tokenizer(examples["document_plaintext"])
        examples["tokens"] = tokens_question + tokens_doc
        examples["labels"] = [0] * len(examples["tokens"])
        examples["length"] = len(examples["labels"])
    else:
        answer_end = answer_start + len(examples["annotations"]["answer_text"][0]) - 1
        before_answer = examples["document_plaintext"][:answer_start]
        answer = examples["annotations"]["answer_text"][0]
        after_answer = examples["document_plaintext"][answer_end + 1 :]
        tokens_question = torch_tokenizer(examples["question_text"]) + ["[SEP]"]
        tokens_before_answer = torch_tokenizer(before_answer)
        tokens_answer = torch_tokenizer(answer)
        tokens_after_answer  = torch_tokenizer(after_answer)
        examples["tokens"] = tokens_question + tokens_before_answer + tokens_answer + tokens_after_answer
        examples["labels"] = [0] * len(tokens_question) + [0] * len(tokens_before_answer) + [1] + [2]*(len(tokens_answer) -1) + [0]*len(tokens_after_answer)
        examples["length"] = len(examples["labels"])
    return examples


tokenized_datasets = english_set.map(tokenizer_data, batched=False,
                                     remove_columns=english_set["train"].column_names)


# Reduce down to our vocabulary and word embeddings
def load_vectors(fname, vocabulary):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    final_vocab = ['[PAD]', '[UNK]', '[BOS]', '[EOS]','[SEP]']
    final_vectors = [np.random.normal(size=(300,)) for _ in range(len(final_vocab))]
    for j,line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        if tokens[0] in vocabulary or len(final_vocab) < 30000:
          final_vocab.append(tokens[0])
          final_vectors.append(np.array(list(map(float, tokens[1:]))))
    return final_vocab, np.vstack(final_vectors)

class FasttextTokenizer:
    def __init__(self, vocabulary):
        self.vocab = {}
        for j,l in enumerate(vocabulary):
            self.vocab[l.strip()] = j

    def encode(self, text):
        # Text is assumed to be tokenized
        return [self.vocab[t] if t in self.vocab else self.vocab['[UNK]'] for t in text]

vocabulary = (set([t for s in tokenized_datasets['train'] for t in s['tokens']]) | set(
    [t for s in tokenized_datasets['validation'] for t in s['tokens']]))
vocabulary, pretrained_embeddings = load_vectors('wiki-news-300d-1M.vec', vocabulary)
print('size of vocabulary: ', len(vocabulary))
tokenizer = FasttextTokenizer(vocabulary)

# Define the model
class BiLSTM(nn.Module):
    """
    Basic BiLSTM-CRF network
    """

    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2
    ):
        """
        Initializer for basic BiLSTM network
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param lstm_dim: The dimensionality of the BiLSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: The number of output classes
        """

        # First thing is to call the superclass initializer
        super(BiLSTM, self).__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a 2 layer BiLSTM, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings,
                                                       padding_idx=pretrained_embeddings.shape[0] - 1),
            'bilstm': nn.LSTM(
                pretrained_embeddings.shape[1],
                lstm_dim,
                2,
                batch_first=True,
                dropout=dropout_prob,
                bidirectional=True),
            'ff': nn.Linear(2 * lstm_dim, n_classes),
        })
        self.n_classes = n_classes
        self.loss = nn.CrossEntropyLoss()
        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['ff'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens, hidden_states=None, labels=None):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :param labels: (b) The label of each sample
        :return: (loss, logits) if `labels` is not None, otherwise just (logits,)
        """

        # Get embeddings (b x sl x edim)
        embeds = self.model['embeddings'](inputs)

        # Pack padded: This is necessary for padded batches input to an RNN
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequence through the BiLSTM
        if hidden_states:
            lstm_out, hidden = self.model['bilstm'](lstm_in, hidden_states)
        else:
            lstm_out, hidden = self.model['bilstm'](lstm_in)

        # Unpack the packed sequence --> (b x sl x 2*lstm_dim)
        lstm_out, hidden_states = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Get logits (b x seq_len x n_classes)
        logits = self.model['ff'](lstm_out)
        outputs = (logits, hidden_states)
        if labels is not None:
            loss = self.loss(logits.reshape(-1, self.n_classes), labels.reshape(-1))
            # log-likelihood from the CRF
            outputs = outputs + (loss,)

        return outputs



class EncoderRNN(nn.Module):
    """
    RNN Encoder model.
    """
    def __init__(self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1):
        """
        Initializer for EncoderRNN network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        """
        # First thing is to call the superclass initializer
        super(EncoderRNN, self).__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, and an LSTM layer.
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'lstm': nn.LSTM(pretrained_embeddings.shape[1], lstm_dim, 2, batch_first=True, bidirectional=True),
        })
        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['lstm'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :return: (lstm output state, lstm hidden state)
        """
        embeds = self.model['embeddings'](inputs)
        lstm_in = nn.utils.rnn.pack_padded_sequence(
                    embeds,
                    input_lens.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
        lstm_out, hidden_states = self.model['lstm'](lstm_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out, hidden_states


class DecoderRNN(nn.Module):
    """
    RNN Decoder model.
    """
    def __init__(self, pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2):
        """
        Initializer for DecoderRNN network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: Number of prediction classes
        """
        # First thing is to call the superclass initializer
        super(DecoderRNN, self).__init__()
        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a LSTM layer, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'lstm': nn.LSTM(pretrained_embeddings.shape[1], lstm_dim, 2, bidirectional=True, batch_first=True),
            'nn': nn.Linear(lstm_dim*2, n_classes),
        })
        # Initialize the weights of the model
        self._init_weights()

    def forward(self, inputs, hidden, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param hidden: (b) The hidden state of the previous step
        :param input_lens: (b) The length of each input sequence
        :return: (output predictions, lstm hidden states) the hidden states will be used as input at the next step
        """
        embeds = self.model['embeddings'](inputs)

        lstm_in = nn.utils.rnn.pack_padded_sequence(
                    embeds,
                    input_lens.cpu(),
                    batch_first=True,
                    enforce_sorted=False
                )
        lstm_out, hidden_states = self.model['lstm'](lstm_in, hidden)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        output = self.model['nn'](lstm_out)
        return output, hidden_states

    def _init_weights(self):
        all_params = list(self.model['lstm'].named_parameters()) + list(self.model['nn'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

# Define the model
class Seq2Seq(nn.Module):
    """
    Basic Seq2Seq network
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            lstm_dim: int,
            dropout_prob: float = 0.1,
            n_classes: int = 2
    ):
        """
        Initializer for basic Seq2Seq network
        :param pretrained_embeddings: A tensor containing the pretrained embeddings
        :param lstm_dim: The dimensionality of the LSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: The number of output classes
        """

        # First thing is to call the superclass initializer
        super(Seq2Seq, self).__init__()

        # We'll define the network in a ModuleDict, which consists of an encoder and a decoder
        self.model = nn.ModuleDict({
            'encoder': EncoderRNN(pretrained_embeddings, lstm_dim, dropout_prob),
            'decoder': DecoderRNN(pretrained_embeddings, lstm_dim, dropout_prob, n_classes),
        })
        self.loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.5]+[5, 2]))


    def forward(self, inputs, input_lens, labels=None):
        """
        Defines how tensors flow through the model.
        For the Seq2Seq model this includes 1) encoding the whole input text,
        and running *target_length* decoding steps to predict the tag of each token.

        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :param labels: (b) The label of each sample
        :return: (loss, logits) if `labels` is not None, otherwise just (logits,)
        """

        # Get embeddings (b x sl x embedding dim)
        encoder_output, encoder_hidden = self.model['encoder'](inputs, input_lens)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([tokenizer.encode(['[BOS]'])]*inputs.shape[0], device=device)
        target_length = labels.size(1)

        loss = None
        for di in range(target_length):
            decoder_output, decoder_hidden = self.model['decoder'](
                decoder_input, decoder_hidden, torch.tensor([1]*inputs.shape[0]))

            if loss == None:
              loss = self.loss(decoder_output.squeeze(1), labels[:, di])
            else:
              loss += self.loss(decoder_output.squeeze(1), labels[:, di])
            # Teacher forcing: Feed the target as the next input
            decoder_input = labels[:, di].unsqueeze(-1)

        return loss / target_length


def train(
        model: nn.Module,
        train_dl: DataLoader,
        valid_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        device: torch.device,
):
    """
    The main training loop which will optimize a given model on a given dataset
    :param model: The model being optimized
    :param train_dl: The training dataset
    :param valid_dl: A validation dataset
    :param optimizer: The optimizer used to update the model parameters
    :param n_epochs: Number of epochs to train for
    :param device: The device to train on
    :return: (model, losses) The best model and the losses per iteration
    """

    # Keep track of the loss and best accuracy
    losses = []
    best_f1 = 0.0

    # Iterate through epochs
    for ep in range(n_epochs):

        loss_epoch = []

        # Iterate through each batch in the dataloader
        for batch in tqdm(train_dl):
            # VERY IMPORTANT: Make sure the model is in training mode, which turns on
            # things like dropout and layer normalization
            model.train()

            # VERY IMPORTANT: zero out all of the gradients on each iteration -- PyTorch
            # keeps track of these dynamically in its computation graph so you need to explicitly
            # zero them out
            optimizer.zero_grad()

            # Place each tensor on the GPU
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            labels = batch[2]
            input_lens = batch[1]

            # Pass the inputs through the model, get the current loss and logits
            loss = model(input_ids, labels=labels, input_lens=input_lens)
            losses.append(loss.item())
            loss_epoch.append(loss.item())

            # Calculate all of the gradients and weight updates for the model
            loss.backward()

            # Optional: clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Finally, update the weights of the model
            optimizer.step()

        # Perform inline evaluation at the end of the epoch
        f1 = evaluate(model, valid_dl)
        print(f'Validation F1: {f1}, train loss: {sum(loss_epoch) / len(loss_epoch)}')

        # Keep track of the best model based on the accuracy
        if f1 > best_f1:
            torch.save(model.state_dict(), 'best_model')
            best_f1 = f1

    return losses


softmax = nn.Softmax(dim=-1)


def decode(model, inputs, input_lens, labels=None, beam_size=2):
    """
    Decoding/predicting the labels for an input text by running beam search.

    :param inputs: (b x sl) The IDs into the vocabulary of the input samples
    :param input_lens: (b) The length of each input sequence
    :param labels: (b) The label of each sample
    :param beam_size: the size of the beam
    :return: predicted sequence of labels
    """

    assert inputs.shape[0] == 1
    # first, encode the input text
    encoder_output, encoder_hidden = model.model['encoder'](inputs, input_lens)
    decoder_hidden = encoder_hidden

    # the decoder starts generating after the Begining of Sentence (BOS) token
    decoder_input = torch.tensor([tokenizer.encode(['[BOS]', ]), ], device=device)
    target_length = labels.shape[1]

    # we will use heapq to keep top best sequences so far sorted in heap_queue
    # these will be sorted by the first item in the tuple
    heap_queue = []
    heap_queue.append((torch.tensor(0), tokenizer.encode(['[BOS]']), decoder_input, decoder_hidden))

    # Beam Decoding
    for _ in range(target_length):
        # print("next len")
        new_items = []
        # for each item on the beam
        for j in range(len(heap_queue)):
            # 1. remove from heap
            score, tokens, decoder_input, decoder_hidden = heapq.heappop(heap_queue)
            # 2. decode one more step
            decoder_output, decoder_hidden = model.model['decoder'](
                decoder_input, decoder_hidden, torch.tensor([1]))
            decoder_output = softmax(decoder_output)
            # 3. get top-k predictions
            best_idx = torch.argsort(decoder_output[0], descending=True)[0]
            # print(decoder_output)
            # print(best_idx)
            for i in range(beam_size):
                decoder_input = torch.tensor([[best_idx[i]]], device=device)

                new_items.append((score + decoder_output[0, 0, best_idx[i]],
                                  tokens + [best_idx[i].item()],
                                  decoder_input,
                                  decoder_hidden))
        # add new sequences to the heap
        for item in new_items:
            # print(item)
            heapq.heappush(heap_queue, item)
        # remove sequences with lowest score (items are sorted in descending order)
        while len(heap_queue) > beam_size:
            heapq.heappop(heap_queue)

    final_sequence = heapq.nlargest(1, heap_queue)[0]
    assert labels.shape[1] == len(final_sequence[1][1:])
    return final_sequence


def evaluate(model: nn.Module, valid_dl: DataLoader, beam_size:int = 1):
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
  logits_all = []
  tags_all = []

  # ALSO IMPORTANT: Don't accumulate gradients during this process
  with torch.no_grad():
    for batch in tqdm(valid_dl, desc='Evaluation'):
      batch = tuple(t.to(device) for t in batch)
      input_ids = batch[0]
      input_lens = batch[1]
      labels = batch[2]

      best_seq = decode(model, input_ids, input_lens, labels=labels, beam_size=beam_size)
      mask = (input_ids != 0)
      labels_all.extend([l for seq,samp in zip(list(labels.detach().cpu().numpy()), input_ids) for l,i in zip(seq,samp) if i != 0])
      tags_all += best_seq[1][1:]
      # print(best_seq[1][1:], labels)
    P, R, F1, _ = precision_recall_fscore_support(labels_all, tags_all, average='macro')
    print(confusion_matrix(labels_all, tags_all))
    return F1

lstm_dim = 300
dropout_prob = 0.1
batch_size = 64
lr = 1e-3
n_epochs = 20

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

# Create the model
model = Seq2Seq(
  pretrained_embeddings=torch.FloatTensor(pretrained_embeddings),
  lstm_dim=lstm_dim,
  dropout_prob=dropout_prob,
  n_classes=3
).to(device)


train_dl = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_batch_bilstm, num_workers=8)
valid_dl = DataLoader(tokenized_datasets['validation'], batch_size=1, collate_fn=collate_batch_bilstm, num_workers=8)

# Create the optimizer
optimizer = Adam(model.parameters(), lr=lr)

# Train
losses = train(model, train_dl, valid_dl, optimizer, n_epochs, device)
model.load_state_dict(torch.load('best_model'))