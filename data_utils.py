'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import os
import sys

import torch

sys.path.append(r'./LAL-Parser/src_joint')
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, T5Tokenizer
from torch.utils.data import Dataset


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                # position
                aspect_post = [aspect['from'], aspect['to']]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


def ParseAugmentData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for slist in data:
            all_data.append(slist)
    return all_data


class Tokenizer4BertGCN:
    def __init__(self, model_name, max_seq_len, pretrained_bert_name):
        global cls, sep, pad
        self.max_seq_len = max_seq_len
        if model_name == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(pretrained_bert_name)
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        elif model_name == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(pretrained_bert_name)

        if model_name == 'bert':
            cls = self.tokenizer.convert_tokens_to_ids('[CLS]')
            sep = self.tokenizer.convert_tokens_to_ids('[SEP]')
            pad = self.tokenizer.convert_tokens_to_ids('[PAD]')
        else:
            cls = self.tokenizer.convert_tokens_to_ids('<s>')
            sep = self.tokenizer.convert_tokens_to_ids('</s>')
            pad = self.tokenizer.convert_tokens_to_ids('<pad>')

        self.cls_token_id = cls
        self.sep_token_id = sep
        self.pad_token_id = pad
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class ABSAData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)  
            offset = len(left) 
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]

            context_asp_len = len(context_asp_ids)

            content_paddings = [tokenizer.pad_token_id] * (tokenizer.max_seq_len - context_asp_len)
            mask_paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)

            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + mask_paddings
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [0] * (len(term_tokens) + 1) + mask_paddings

            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + mask_paddings
            context_asp_ids += content_paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ABSAFinetuneData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Fine-tuning examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            # context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
            #     bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [
            #                       tokenizer.sep_token_id]
            context_asp_ids = tokenizer.convert_tokens_to_ids(bert_tokens) + [tokenizer.sep_token_id]  # delete aspect concat and cls
            context_asp_len = len(context_asp_ids)

            content_paddings = [tokenizer.pad_token_id] * (tokenizer.max_seq_len - context_asp_len)
            mask_paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)

            # context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + mask_paddings
            context_asp_seg_ids = [0] * tokenizer.max_seq_len

            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + mask_paddings
            context_asp_ids += content_paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ABSAGenerateData(Dataset):
    def __init__(self, fname, tokenizer, opt):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Generation examples"):
            polarity = polarity_dict[obj['label']]
            text = obj['text']
            term = obj['aspect']
            term_start = obj['aspect_post'][0]
            term_end = obj['aspect_post'][1]
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens
            # context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
            #     bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [
            #                       tokenizer.sep_token_id]
            context_asp_ids = tokenizer.convert_tokens_to_ids(bert_tokens)  # delete aspect concat and cls
            context_asp_len = len(context_asp_ids)

            content_paddings = [tokenizer.pad_token_id] * (tokenizer.max_seq_len - context_asp_len)
            mask_paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)

            # context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + mask_paddings
            context_asp_seg_ids = [0] * tokenizer.max_seq_len

            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + mask_paddings
            # context_asp_ids += content_paddings  # generateSet have not paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,
                'aspect_mask': aspect_mask,
                'polarity': polarity,
            }
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ABSAAugmentData(Dataset):
    def __init__(self, fname, tokenizer):
        self.data = []
        parse = ParseAugmentData

        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Augmentation examples"):
            data = []
            for sen in obj:
                bert_tokens = tokenizer.tokenize(sen)
                context_asp_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
                # context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
                context_asp_ids = torch.tensor(context_asp_ids)
                data.append(context_asp_ids)
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # return a 2-d list