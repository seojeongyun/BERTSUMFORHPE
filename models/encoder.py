import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention, PositionwiseFeedForward


class Classifier(nn.Module):
    def __init__(self, hidden_size, args):
        super(Classifier, self).__init__()
        # ver 1
        self.args = args
        if self.args.decouple_mode == 'Shared':
            self.shared_1 = nn.Linear(hidden_size, 512)
            self.shared_2 = nn.Linear(512, 256)


        elif self.args.decouple_mode == 'Full':
            self.exercise_linear = nn.Linear(hidden_size, 256)

            self.conditions_linear_1 = nn.Linear(hidden_size, 1024)
            self.conditions_linear_2 = nn.Linear(1024, 512)
            self.conditions_linear_3 = nn.Linear(512, 256)

        else:
            raise ValueError(f"Unknown decouple_mode: {self.args.decouple_mode}")
        #
        self.exercise_classifier =nn.Linear(256 ,41)
        self.conditions_classifier = nn.Linear(256, 97)
        #

        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # [BS,SEQ_LEN,DIM]
        cls = x[:,0,:]
        if self.args.decouple_mode == 'Full':
            exercise_out = self.exercise_linear(cls)
            pred_exercise = self.exercise_classifier(self.act(exercise_out))
            #
            condition_out = self.conditions_linear_1(cls)
            condition_out = self.conditions_linear_2(self.act(condition_out))
            condition_out = self.conditions_linear_3(self.act(condition_out))
            pred_conditions = self.conditions_classifier(condition_out)

        else:
            shared_feat = self.shared_1(cls)
            shared_feat = self.shared_2(self.act(shared_feat))
            pred_exercise = self.exercise_classifier(self.act(shared_feat))
            pred_conditions = self.conditions_classifier(shared_feat)

        #
        return pred_exercise, pred_conditions


class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = (emb * math.sqrt(self.dim)).unsqueeze(0)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)].to(emb.device)
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, 1 - mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores

