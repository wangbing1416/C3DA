import os
import sys
import copy
import random
import logging
import heapq

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import AdamW
from models.predictModel import PredictModel
from data_utils import Tokenizer4BertGCN, ABSAData, ABSAAugmentData
from CLoss import ContrastiveLoss
from prepare_vocab import VocabHelp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    return list(batch)

class Instructor:
    ''' Model training and evaluation '''
    def __init__(self, opt):
        self.opt = opt

        tokenizer = Tokenizer4BertGCN(opt.model_name, opt.max_length, opt.pretrained_bert_name)

        self.model = PredictModel(opt).to(opt.device)
        trainset = ABSAData(opt.dataset_file['train'], tokenizer, opt=opt)
        testset = ABSAData(opt.dataset_file['test'], tokenizer, opt=opt)
        augset = ABSAAugmentData(opt.dataset_file['aug'], tokenizer)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)
        # TODO: dataloader need a collate_fn to output special data structure √
        self.aug_dataloader = DataLoader(dataset=augset, batch_size=opt.batch_size, collate_fn=collate_fn, shuffle=False)

        self.closs = ContrastiveLoss(batch_size=opt.batch_size, margin=opt.margin)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    
    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')
        
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
    
    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)   # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0]**0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                            any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def entropy_filter(self, aug_inputs):
        self.model.eval()
        aug_filter = []
        for group in aug_inputs:
            # entropy = 10  # max entropy is near to 2?
            entropy = []
            for sen in group:
                sen = sen.to(self.opt.device)
                if sen.shape[0] == 0:
                    sen = torch.tensor([3]).to(self.opt.device)  # avoid empty generation
                logit = F.softmax(self.model(sen.unsqueeze(dim=0), mode='filter'), dim=1)
                # entropy larger, more stable, so we want a mix-entropy sample
                entropy.append(- torch.sum(logit * torch.log2(logit)))
            aug_filter.append([])
            for k in heapq.nlargest(self.opt.k, range(len(entropy)), entropy.__getitem__):
                aug_filter[-1].append(group[k])
        return aug_filter


    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        global loss_aug, loss_cl
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0

            vanilla_iter = iter(self.train_dataloader)
            aug_iter = iter(self.aug_dataloader)

            for _ in range(self.train_dataloader.__len__()):
                vanilla_sample = vanilla_iter.next()
                aug_sample = aug_iter.next()
                global_step += 1
                aug_inputs = self.entropy_filter(aug_sample)
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()

                # training objective!
                inputs = [vanilla_sample[col].to(self.opt.device) for col in self.opt.inputs_cols]
                feature, outputs, _ = self.model(inputs, mode='trainandeval')
                targets = vanilla_sample['polarity'].to(self.opt.device)
                loss_van = criterion(outputs, targets)
                loss = criterion(outputs, targets)

                if self.opt.withAugment:
                    inputs_copy = copy.deepcopy(inputs)
                    targets_copy = copy.deepcopy(targets)
                    for _ in range(self.opt.k - 1):
                        for i in range(len(inputs_copy)):
                            inputs_copy[i] = torch.cat([inputs_copy[i], copy.deepcopy(inputs)[i]], dim=0)
                        targets_copy = torch.cat([targets_copy, copy.deepcopy(targets)], dim=0)

                    for k in range(inputs_copy[0].shape[0]):
                        start = torch.sum(inputs_copy[2][k])
                        len_rem = self.opt.max_length - start
                        front = k % inputs[0].shape[0]
                        back = k // inputs[0].shape[0]
                        if len_rem >= aug_inputs[front][back].shape[0]:  # avoid sentences is so long
                            inputs_copy[0][k][start: start + aug_inputs[front][back].shape[0]] = aug_inputs[front][back].to(self.opt.device)
                        else:
                            inputs_copy[0][k][start: self.opt.max_length] = aug_inputs[front][back][:len_rem].to(self.opt.device)
                    feature_aug, outputs_aug, feature_last = self.model(inputs_copy, mode='trainandeval')
                    # targets = vanilla_sample['polarity'].to(self.opt.device)
                    loss_aug = criterion(outputs_aug, targets_copy)
                    loss += loss_aug * self.opt.aug_loss_fac

                    if self.opt.withCL:  # if 'withAugment' is False, 'withCL' will not be executed
                        feature_clone = feature.clone()
                        for _ in range(self.opt.k - 1):
                            feature = torch.cat([feature, feature_clone], dim=0)
                        loss_cl = self.closs(feature, feature_aug, feature_last)
                        loss += loss_cl * self.opt.cl_loss_fac

                loss.backward()
                optimizer.step()
                
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./C3DA/state_dict'):
                                os.mkdir('./C3DA/state_dict')
                            model_path = './C3DA/state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    if self.opt.withAugment:
                        if self.opt.withCL:
                            logger.info('loss: {:.4f}, VanillaLoss:{:.4f}, AugLoss:{:.4f}, CLLoss:{:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'
                                        .format(loss.item(), loss_van.item(), loss_aug.item(), loss_cl.item(), train_acc, test_acc, f1))
                        else:
                            logger.info('loss: {:.4f}, VanillaLoss:{:.4f}, AugLoss:{:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'
                                .format(loss.item(), loss_van.item(), loss_aug.item(), train_acc, test_acc, f1))
                    else:
                        logger.info('loss: {:.4f}, VanillaLoss:{:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), loss_van.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path
    
    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                _, outputs, _ = self.model(inputs, mode='trainandeval')
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)
        
    
    def run(self):
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        'bartencoder': 'facebook/bart-base',
    }
    

    
    input_colses = {
        'bert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask', 'aspect_mask'],
        'roberta': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                 'aspect_mask'],
        'bart':['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end', 'src_mask',
                 'aspect_mask']
    }
    
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    
    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad, 
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax, 
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    vocab_dirs = {
        'restaurant': './dataset/Restaurants_corenlp/',
        'laptop': './dataset/Laptops_corenlp/',
        'twitter': './dataset/Tweets_corenlp/',
    }
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help='restaurant, laptop, twitter')
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))

    parser.add_argument('--generate_model', default='t5', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--prompt_name', default='lora', type=str, help='prompttuning, prefixtuning, adapter, lora')
    parser.add_argument('--ft_epoch', default=100, type=int)

    parser.add_argument('--aug_loss_fac', default=1.0, type=float)
    parser.add_argument('--margin', default=0.3, type=float)
    parser.add_argument('--cl_loss_fac', default=2, type=float)
    parser.add_argument('--withAugment', default=False, action='store_true')
    parser.add_argument('--withCL', default=False, action='store_true')
    parser.add_argument('--k', default=1, type=int)
    # parser.add_argument('--withAugment', default=True)
    # parser.add_argument('--withCL', default=True)

    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=15, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)

    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')

    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--max_length', default=120, type=int)
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

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
            'aug': './dataset/Restaurants_corenlp/generate-{}-{}-{}.json'.format(opt.generate_model, opt.prompt_name, opt.ft_epoch),
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json',
            'aug': './dataset/Laptops_corenlp/generate-{}-{}-{}.json'.format(opt.generate_model, opt.prompt_name, opt.ft_epoch),
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
            'aug': './dataset/Tweets_corenlp/generate-{}-{}-{}.json'.format(opt.generate_model, opt.prompt_name, opt.ft_epoch),
        }
    }

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.vocab_dir = vocab_dirs[opt.dataset]
    opt.pretrained_bert_name = model_classes[opt.model_name]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(opt.device)
    
    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./C3DA/log'):
        os.makedirs('./C3DA/log', mode=0o777)
    log_file = '{}-{}-s{}-{}.log'.format(opt.model_name, opt.dataset, opt.seed, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./C3DA/log', log_file)))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
