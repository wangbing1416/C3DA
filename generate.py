import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import copy
import random
import logging
import json
import collections
import math

import argparse
import torch
import torch.nn as nn

import numpy as np
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor
from models.Adapter import AdapterModel
from models.SoftPrompt import SoftPromptModel
from models.LoRa import LoraModel
from models.Prefix import PrefixModel

from data_utils import Tokenizer4BertGCN, ABSAGenerateData, ABSAFinetuneData

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        global beta
        self.opt = opt

        self.tokenizer = Tokenizer4BertGCN(opt.model_name, opt.max_length, opt.model_class)
        self.backbone = T5ForConditionalGeneration.from_pretrained(opt.model_class).to(opt.device)

        if opt.prompt_name == 'none':
            self.model = self.backbone
        else:
            if opt.prompt_name == 'prompttuning':
                self.model = SoftPromptModel(backbone_model=self.backbone, soft_token_num=opt.soft_token_num)
            # TODO:debug the source code in opendelta, such as adding .to('cuda') to other prompts
            elif opt.prompt_name == 'adapter':
                self.model = AdapterModel(backbone_model=self.backbone)
            elif opt.prompt_name == 'prefixtuning':
                self.model = PrefixModel(backbone_model=self.backbone)
            else: # lora
                self.model = LoraModel(backbone_model=self.backbone, lora_dropout=opt.lora_dropout)
            self.model.freeze_module(exclude=["deltas", "layernorm_embedding"], set_state_dict=True)

        finetuneSet = ABSAFinetuneData(opt.dataset_file['train'], self.tokenizer, opt=opt)
        generateSet = ABSAGenerateData(opt.dataset_file['train'], self.tokenizer, opt=opt)
        # testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)

        # fine-tune stage can utilize shuffle
        self.finetune_dataloader = DataLoader(dataset=finetuneSet, batch_size=opt.batch_size, shuffle=True,
                                              drop_last=True, num_workers=4)
        self.generate_dataloader = DataLoader(dataset=generateSet, batch_size=1, shuffle=False, num_workers=4)
        # self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        self.polarity_dic = {  # tensor
            0: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('</s> so good'))),
            1: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('</s> so bad'))),
            2: torch.tensor(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('</s> and')))
        }

        counter = collections.Counter()
        M = 0
        for sam in finetuneSet.data:
            counter += collections.Counter(sam['text_bert_indices'][sam['asp_start']: sam['asp_end']])
            M += sam['asp_end'] - sam['asp_start']
        C = math.log(M - 1) * math.pow((opt.B + 1), opt.A)
        self.beta_dic = dict()
        for asp in counter:
            beta = 1 + C * math.exp(- opt.A * math.log(counter[asp] + opt.B))
            self.beta_dic[asp] = beta
        for item in self.beta_dic:
            self.beta_dic[item] = self.beta_dic[item] / beta

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))


    def run(self):
        # fine-tune
        logger.info('Fine-tuning Prompt Stage...')
        global_step = 0
        optimizer = Adafactor(params=self.backbone.parameters(),relative_step=True, warmup_init=True, lr=None)

        for epoch in range(self.opt.num_epoch):

            for i_batch, sample_batched in enumerate(self.finetune_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.backbone.train()
                optimizer.zero_grad()

                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]

                aspect_input_ids = inputs[0].clone()
                polar_input_ids = inputs[0].clone()
                # prepare labels
                label_ids = inputs[0].clone()
                temp = label_ids[0].clone()  # we fix a bug, thank Jie Shen@Fudan
                label_ids[:-1] = label_ids[1:].clone()
                label_ids[-1] = temp
                labels = torch.cat([label_ids, label_ids], dim=0)
                # prepare fine-tune instances / aspect fine-tune
                for i in range(aspect_input_ids.shape[0]):
                    j = (i + 1) % aspect_input_ids.shape[0]
                    start = torch.sum(inputs[2][i])
                    aspect_input_ids[i][start] = torch.tensor(self.tokenizer.convert_tokens_to_ids('</s>')).to(self.opt.device)
                    start += 1
                    aspect_input_ids[i][start: start + inputs[4][j] - inputs[3][j]] = aspect_input_ids[j][inputs[3][j]: inputs[4][j]]

                # prepare fine-tune instances / polarity fine-tune
                for i in range(polar_input_ids.shape[0]):
                    j = (i + 1) % polar_input_ids.shape[0]
                    start = torch.sum(inputs[2][i])
                    polarLen = self.polarity_dic[inputs[-1][j].item()].shape[0]
                    polar_input_ids[i][start: start + polarLen] = self.polarity_dic[inputs[-1][j].item()]
                input_ids = torch.cat([aspect_input_ids, polar_input_ids], dim=0)

                outputs = self.backbone(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # instance-level loss clip
                # loss = None
                # den = 0
                # for i in range(aspect_input_ids.shape[0]):
                #     k = (i + 1) % self.opt.batch_size
                #     outputs = self.backbone(input_ids=aspect_input_ids[i].unsqueeze(dim=0), labels=label_ids[i].unsqueeze(dim=0))
                #     beta = 0
                #     for token in inputs[0][k][inputs[3][k]: inputs[4][k]]:
                #         beta += self.beta_dic[token.item()]
                #     if i == 0: loss = outputs.loss * beta / (inputs[4][k] - inputs[3][k])
                #     else: loss += outputs.loss * beta / (inputs[4][k] - inputs[3][k])
                #     den += beta
                # loss /= den
                #
                # outputs = self.backbone(input_ids=polar_input_ids, labels=label_ids)
                # loss += outputs.loss

                loss.backward()
                optimizer.step()

                if epoch % 10 == 0 and i_batch == 0:
                    # Case Study 1
                    input_ids = inputs[0][0]
                    logger.info("Case Study 1 -> Source Sentence:")
                    logger.info(self.tokenizer.decode(input_ids))
                    generation_ids = self.backbone.generate(input_ids.unsqueeze(dim=0))
                    logger.info("Case Study 1 -> Target Sentence:")
                    logger.info(self.tokenizer.decode(generation_ids[0]))
                    # Case Study 2
                    input_ids = inputs[0][1]
                    logger.info("Case Study 2 -> Source Sentence:")
                    logger.info(self.tokenizer.decode(input_ids))
                    generation_ids = self.backbone.generate(input_ids.unsqueeze(dim=0))
                    logger.info("Case Study 2 -> Target Sentence:")
                    logger.info(self.tokenizer.decode(generation_ids[0]))
            logger.info("Fine-tune-epoch:{}, Fine-tuneLoss:{:4f}".format(epoch, loss.item()))
        logger.info("*** Fine-tune Stage Finished! ***")

        # generation
        logger.info("Generation Stage...")
        generation_data = []
        for i_batch, sample_batched in enumerate(self.generate_dataloader):
            self.backbone.eval()
            inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
            # TODO: check if it is no paddings! check source_sencence 2022-03-11
            input_ids = inputs[0]  # tensor: batch size = 1 * max length (no paddings)

            # once generation
            source_sentence_acc1 = self.AspectAugentationChannel(input_ids)
            # num_beam : beam search
            target_generation_acc1 = self.backbone.generate(source_sentence_acc1, max_length=self.opt.generate_max_len, min_length=self.opt.generate_min_len, num_beams=1)
            source_sentence_pcc1 = self.PolarityAugmentationChannel(input_ids, self.opt.polarity_dim - inputs[-1] - 1)
            target_generation_pcc1 = self.backbone.generate(source_sentence_pcc1, max_length=self.opt.generate_max_len, min_length=self.opt.generate_min_len, num_beams=1)

            # twice generation
            source_sentence_acc2 = self.AspectAugentationChannel(target_generation_pcc1)
            target_generation_acc2 = self.backbone.generate(source_sentence_acc2, max_length=self.opt.generate_max_len, min_length=self.opt.generate_min_len, num_beams=1)
            source_sentence_pcc2 = self.PolarityAugmentationChannel(target_generation_acc1, self.opt.polarity_dim - inputs[-1] - 1)
            target_generation_pcc2 = self.backbone.generate(source_sentence_pcc2, max_length=self.opt.generate_max_len, min_length=self.opt.generate_min_len, num_beams=1)

            # index to sentence tokens
            target_generation_acc1_tokens = self.tokenizer.decode(target_generation_acc1[0])
            target_generation_pcc1_tokens = self.tokenizer.decode(target_generation_pcc1[0])
            target_generation_acc2_tokens = self.tokenizer.decode(target_generation_acc2[0])
            target_generation_pcc2_tokens = self.tokenizer.decode(target_generation_pcc2[0])
            generation_data.append([target_generation_acc1_tokens, target_generation_pcc1_tokens,
                                    target_generation_acc2_tokens, target_generation_pcc2_tokens])
            if i_batch % 50 == 0:
                logger.info('{} samples have been generated...'.format(i_batch))

        # save document
        try:
            with open('{}generate-{}-{}-{}.json'.format(self.opt.vocab_dir, self.opt.model_name, self.opt.prompt_name, self.opt.num_epoch), 'w') as file:
                json.dump(generation_data, fp=file, indent=6)
        except IOError as error:
            logger.info(error)
        logger.info('*** Generated Data Items have been saved! ***')

        logger.info('#' * 60)

    def AspectAugentationChannel(self, input_ids):
        # TODO: add some multi_words aspects
        aspect_dic = {  # 30 aspects for each domain
            'restaurant': ['staff', 'food', 'kitchen', 'menu', 'perks', 'waiters', 'meats', 'dish', 'cheese', 'plate',
                           'drinks', 'design', 'atmosphere', 'pizza', 'seats', 'eat family style', 'service', 'price',
                           'wine', 'quantity', 'sushi', 'fried rice', 'salad', 'indian food', 'broth with noodles',
                           'money', 'thai food', 'glass of wine', 'lunch', 'dinner'],
            'laptop':['cord', 'battery life', 'service center', 'tech guy', 'quality', 'applications', 'use',
                      'start up', 'features', 'garage band', 'screen', 'power light', 'hard driver light',
                      'processor', 'graphics cards', 'rubber enclosure', 'tracking area', 'external mouse',
                      'suite of software', 'speed', 'windows 7', 'usb devices', 'system', 'boot up', 'warranty service',
                      'force quit', 'navigate', 'keys', 'images', 'switch'],
            'twitter':['google', 'harry potter', 'lady gaga', 'george bush', 'iphone', 'wave', 'britney spears',
                       'kindle', 'wii', 'windows 7', 'charlie sheen', 'ipod', 'lindsay lohan', 'justin bieber',
                       'xbox', 'lakers', 'bill gates', 'madonna', ]
        }

        # TODO: this template can be optimized, such as 'and+aspect' ',+aspect', etc. in addition ',' is always noisy
        asp_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(['</s>'] +
            self.tokenizer.tokenize(random.choice(aspect_dic[self.opt.dataset])))).unsqueeze(dim=0).to(self.opt.device)
        return torch.cat([input_ids, asp_ids],dim=1)


    def PolarityAugmentationChannel(self, input_ids, polar):
        polar_ids = self.polarity_dic[polar.item()].unsqueeze(dim=0).to(self.opt.device)
        return torch.cat([input_ids, polar_ids], dim=1)



def main():
    model_classes = {
        't5': 'mrm8488/t5-base-finetuned-common_gen',
        # 't5': 't5-base',
        'gpt': 'gpt-2',
    }

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }
    }

    input_colses = {
        'bart': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                 'aspect_mask', 'polarity'],
        't5': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                    'aspect_mask', 'polarity']
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    vocab_dirs = {
        'restaurant': './dataset/Restaurants_corenlp/',
        'laptop': './dataset/Laptops_corenlp/',
        'twitter': './dataset/Tweets_corenlp/',
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='t5', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--prompt_name', default='none', type=str, help='prompttuning, prefixtuning, adapter, lora')
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))

    parser.add_argument('--l2reg', default=1e-4, type=float)
    # num_epoch = 0 -> have no fine-tune stage
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)

    parser.add_argument('--A', default=2, type=int)
    parser.add_argument('--B', default=2, type=int)
    parser.add_argument('--soft_token_num', default=100, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)



    parser.add_argument('--prompt_len', default=3, type=int, help='soft prompt length')
    parser.add_argument('--polarity_dim', default=3, type=int, help='3')
    parser.add_argument('--generate_max_len', default=50, type=int, help='max length of generated sentences')
    parser.add_argument('--generate_min_len', default=10, type=int, help='min length of generated sentences')

    parser.add_argument('--num_aug_step1', default=1, type=int)
    parser.add_argument('--num_aug_step2', default=1, type=int)

    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--device', default='cuda', type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--cuda', default='1', type=str)

    # * bert
    # parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.vocab_dir = vocab_dirs[opt.dataset]
    opt.pretrained_bert_name = model_classes[opt.model_name]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./C3DA/generate/log'):
        os.makedirs('./C3DA/generate/log', mode=0o777)
    log_file = '{}-{}-{}-{}.log'.format(opt.model_name, opt.prompt_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./C3DA/generate/log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
