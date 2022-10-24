import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification


class attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop):
        super(attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print(ntype+" mp ", beta.data.cpu().numpy())  # semantic attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp = z_mp + embeds[i] * beta[i]
        return z_mp


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 32)
        self.layer3 = nn.Linear(32, output_dim)

    def forward(self, data):
        data = F.relu(self.layer1(data))
        data = F.relu(self.layer2(data))
        data = self.layer3(data)
        return data


def mean_pooling(model_output, attention_mask):
    # Mean Pooling - Take attention mask into account for correct averaging
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
    sum_embeddings = torch.sum(model_output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class QA_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QA_model, self).__init__()
        self.que_in_mlp = nn.Linear(input_dim, hidden_dim)
        self.context_in_mlp = nn.Linear(input_dim, hidden_dim)
        self.output_layer = MLP(input_dim, hidden_dim, output_dim)
        self.attention_layer = attention(hidden_dim, 0.5)
        model_checkpoint = "bert-base-multilingual-uncased"
        self.transformer_layer = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    def forward(self, input):
        input_ids = torch.stack(input["input_ids"], dim=1).cuda()
        attention_mask = torch.stack(input["attention_mask"], dim=1).cuda()
        # question_input_ids = torch.stack(input['question_input_ids'],dim=0).cuda()
        # question_attention_mask=torch.stack(input['question_attention_mask'],dim=0).cuda()
        # document_input_ids = torch.stack(input['document_input_ids'],dim=0).cuda()
        # document_attention_mask = torch.stack(input['document_attention_mask'],dim=0).cuda()
        hidden_states = \
        self.transformer_layer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                               return_dict=True)['hidden_states']
        hidden_out = mean_pooling(hidden_states[-1], attention_mask)
        predict_label = F.sigmoid(self.output_layer(hidden_out))

        return predict_label
