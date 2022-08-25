import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel, AdamW, RobertaModel


class PredictModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(opt.pretrained_bert_name)

        self.opt = opt
        self.classifier = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs, mode):

        if mode == 'trainandeval':  # aspect representation for classification
            text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, src_mask, aspect_mask = inputs
            res = self.backbone(text_bert_indices)  # del attention_mask and bert_segments_ids

            sequence_ouput = res[0]
            # pooled_output = res[1]

            asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
            aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
            aspect_mean = (sequence_ouput * aspect_mask).sum(dim=1) / asp_wn

            logits = self.classifier(aspect_mean)
            return aspect_mean, logits, sequence_ouput[:, -1, :].squeeze()

        else:  # sentence representation for filter
            res = self.backbone(inputs)  # del attention_mask and bert_segments_ids

            # sequence_ouput = res[0]
            pooled_output = res[1]

            logits = self.classifier(pooled_output)

            return logits
