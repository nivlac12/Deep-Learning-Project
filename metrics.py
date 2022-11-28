import numpy as np

import json
from os.path import join
import torch
import logging
import tempfile
import subprocess as sp
from datetime import timedelta
from time import time
from itertools import combinations

from pyrouge import Rouge155
from pyrouge.utils import log

import tensorflow as tf

#python pkg - should work for both tf and pytorch provided that
#it fits with the expected datatype/struct of tf
#https://pypi.org/project/rouge/
from rouge import Rouge

#FastNLP is a modular Natural Language Processing system based on PyTorch, built for fast development of NLP models.
#note: from the pypi page, doesn't appear to to be implemented in tf
#also, the pypi page is in chinese lol, so you are better off reading the
#document below
#https://readthedocs.org/projects/zyfeng-fastnlp/downloads/pdf/v0.3.0/
#LossBase is the base class for all losses
from fastNLP.core.losses import LossBase

# MetricBase handles validity check of its input dictionaries - pred_dict and target_dict.
# pred_dict is the output of forward() or prediction function of a model. target_dict is the ground
# truth from DataSet where is_target is set True. MetricBase will do the following type checks:
# 1. whether self.evaluate has varargs, which is not supported.
# 3.1. fastNLP 19
# fastNLP Documentation, Release 0.2
# 2. whether params needed by self.evaluate is not included in pred_dict, target_dict.
# 3. whether params needed by self.evaluate duplicate in pred_dict, target_dict.
# 4. whether params in pred_dict, target_dict are not used by evaluate.(Might cause warning)
# Besides, before passing params into self.evaluate, this function will filter out params from output_dict and
# target_dict which are not used in self.evaluate. (but if **kwargs presented in self.evaluate, no filtering will be
# conducted.) However, in some cases where type check is not necessary, _fast_param_map will be used.
from fastNLP.core.metrics import MetricBase

_ROUGE_PATH = '/path/to/RELEASE-1.5.5'

def loss(margin):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

    Arguments:
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        'constrastive_loss' function with data ('margin') attached.
    """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(x1, x2,y):
        """Calculates the constrastive loss.

        Arguments:
            y_true: List of labels, each label is of type float32.
            y_pred: List of predictions of same length as of y_true,
                    each label is of type float32.

        Returns:
            A tensor containing constrastive loss as floating point value.
        """
        #is this supposed to be a dot so that margin can be a scalar?
        my_loss = tf.math.maximum(0,-y*(x1-x2)+margin)
        return my_loss

    return contrastive_loss

#LossBase, the base class for all losses defined in the fastNLP package,
#serves as the parent class
class MarginRankingLoss(LossBase):      
    
    def __init__(self, margin, score=None, summary_score=None):
        #I think __init__ here pulls the vars in the existing LossBase pkg
        #?
        super(MarginRankingLoss, self).__init__()
        self._init_param_map(score=score, summary_score=summary_score)

        #assigns to self var so var initialzed can be used elsewhere
        self.margin = margin

        #pytorch version of layers for this type of loss


    def get_loss(self, score, summary_score):
        
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        #Returns a tensor filled with the scalar value 1, with the shape defined 
        # by the variable argument size.


        ones = tf.ones(score.shape()).cuda(score.device)

        #link below details the equations
        #https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html
        #loss_func = torch.nn.MarginRankingLoss(0.0)
        loss_func = loss(0.0)

        TotalLoss = loss_func(0,score, score, ones)

        # candidate loss
        n = score.shape(1)
        for i in range(1, n):
            #includes all but i last vals in seq
            pos_score = score[:, :-i]

            #includes only first i values in 2nd dim of seq
            neg_score = score[:, i:]
            pos_score = pos_score.reshape(-1)
            neg_score = neg_score.reshape(-1)
            ones = tf.ones(pos_score.shape()).cuda(score.device)
            # loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            loss_func = loss(self.margin * i)

            #seems to add this new loss_fun value to a list of loss
            #values
            #used self.margin*i to mimic initial implementation
            TotalLoss += loss_func(pos_score, neg_score, ones)

        # gold summary loss
        pos_score = tf.expand_dims(summary_score,-1).broadcast_to(score.shape)
        neg_score = score
        pos_score = pos_score.reshape(-1)
        neg_score = neg_score.reshape(-1)
        ones = tf.ones(pos_score.shape()).cuda(score.device)
        # loss_func = torch.nn.MarginRankingLoss(0.0)
        loss_func = loss(0.0)

        TotalLoss += loss_func(pos_score, neg_score, ones)
        
        return TotalLoss

class ValidMetric(MetricBase):
    def __init__(self, save_path, data, score=None):
        super(ValidMetric, self).__init__()
        self._init_param_map(score=score)
 
        self.save_path = save_path
        self.data = data

        self.top1_correct = 0
        self.top6_correct = 0
        self.top10_correct = 0
         
        #create instance of Rouge from imported pkg
        self.rouge = Rouge()
        self.ROUGE = 0.0
        self.Error = 0

        self.cur_idx = 0
    
    # an approximate method of calculating ROUGE
    def fast_rouge(self, dec, ref):
        if dec == '' or ref == '':
            return 0.0
        
        #get rouge scoesr using dec and ref
        scores = self.rouge.get_scores(dec, ref)

        #seems to return rouge-1*2 + rouge-2/3 but I'm not sure
        return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

    def evaluate(self, score):
        batch_size = score.shape(0)
        #torch.max: Returns the maximum value of all elements in the 
        # input tensor. does along 1st dim
        #tf.math.reduce_sum: Returns the sum of all elements in the input tensor.
        #converst all of that to an int
        #then appends that calculated val to the list
        #indices seems to be pulling the max values within that set of indices
        #tf.math.reduce_max computes tf.math.maximum of elements across dimensions of a tensor

        self.top1_correct += int(tf.math.reduce_sum(tf.math.reduce_max(score, axis=1).indices == 0))
        self.top6_correct += int(tf.math.reduce_sum(tf.math.reduce_max(score, axis=1).indices <= 5))
        self.top10_correct += int(tf.math.reduce_sum(tf.math.reduce_max(score, axis=1).indices <= 9))

        # Fast ROUGE
        for i in range(batch_size):
            max_idx = int(tf.math.reduce_max(score[i], axis=0).indices)
            if max_idx >= len(self.data[self.cur_idx]['indices']):
                self.Error += 1 # Check if the candidate summary generated by padding is selected
                self.cur_idx += 1
                continue
            ext_idx = self.data[self.cur_idx]['indices'][max_idx]
            ext_idx.sort()
            dec = []
            ref = ' '.join(self.data[self.cur_idx]['summary'])
            for j in ext_idx:
                dec.append(self.data[self.cur_idx]['text'][j])
            dec = ' '.join(dec)
            self.ROUGE += self.fast_rouge(dec, ref)
            self.cur_idx += 1

    def get_metric(self, reset=True):
        top1_accuracy = self.top1_correct / self.cur_idx
        top6_accuracy = self.top6_correct / self.cur_idx
        top10_accuracy = self.top10_correct / self.cur_idx
        ROUGE = self.ROUGE / self.cur_idx
        eval_result = {'top1_accuracy': top1_accuracy, 'top6_accuracy': top6_accuracy, 
                       'top10_accuracy': top10_accuracy, 'Error': self.Error, 'ROUGE': ROUGE}
        with open(join(self.save_path, 'train_info.txt'), 'a') as f:
            print('top1_accuracy = {}, top6_accuracy = {}, top10_accuracy = {}, Error = {}, ROUGE = {}'.format(
                  top1_accuracy, top6_accuracy, top10_accuracy, self.Error, ROUGE), file=f)
        if reset:
            self.top1_correct = 0
            self.top6_correct = 0
            self.top10_correct = 0
            self.ROUGE = 0.0
            self.Error = 0
            self.cur_idx = 0
        return eval_result
        
class MatchRougeMetric(MetricBase):
    def __init__(self, data, dec_path, ref_path, n_total, score=None):
        super(MatchRougeMetric, self).__init__()
        self._init_param_map(score=score)
        self.data        = data
        self.dec_path    = dec_path
        self.ref_path    = ref_path
        self.n_total     = n_total
        self.cur_idx = 0
        self.ext = []
        self.start = time()

    
    def evaluate(self, score):
        ext = int(tf.math.reduce_max(score, axis=1).indices) # batch_size = 1
        self.ext.append(ext)
        self.cur_idx += 1
        print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
              self.cur_idx, self.n_total, self.cur_idx/self.n_total*100, timedelta(seconds=int(time()-self.start))
             ), end='')
    
    def get_metric(self, reset=True):
        
        print('\nStart writing files !!!')
        for i, ext in enumerate(self.ext):
            sent_ids = self.data[i]['indices'][ext]
            dec, ref = [], []
            
            for j in sent_ids:
                dec.append(self.data[i]['text'][j])
            for sent in self.data[i]['summary']:
                ref.append(sent)

            with open(join(self.dec_path, '{}.dec'.format(i)), 'w') as f:
                for sent in dec:
                    print(sent, file=f)
            with open(join(self.ref_path, '{}.ref'.format(i)), 'w') as f:
                for sent in ref:
                    print(sent, file=f)
        
        print('Start evaluating ROUGE score !!!')
        R_1, R_2, R_L = MatchRougeMetric.eval_rouge(self.dec_path, self.ref_path)
        eval_result = {'ROUGE-1': R_1, 'ROUGE-2': R_2, 'ROUGE-L':R_L}

        if reset == True:
            self.cur_idx = 0
            self.ext = []
            self.data = []
            self.start = time()
        return eval_result
        
    @staticmethod
    def eval_rouge(dec_dir, ref_dir, Print=True):
        assert _ROUGE_PATH is not None

        #set logger to output warning if smt arises to level of warning
        log.get_global_console_logger().setLevel(logging.WARNING)
        dec_pattern = '(\d+).dec'
        ref_pattern = '#ID#.ref'
        cmd = '-c 95 -r 1000 -n 2 -m'
        with tempfile.TemporaryDirectory() as tmp_dir:
            Rouge155.convert_summaries_to_rouge_format(
                dec_dir, join(tmp_dir, 'dec'))
            Rouge155.convert_summaries_to_rouge_format(
                ref_dir, join(tmp_dir, 'ref'))
            Rouge155.write_config_static(
                join(tmp_dir, 'dec'), dec_pattern,
                join(tmp_dir, 'ref'), ref_pattern,
                join(tmp_dir, 'settings.xml'), system_id=1
            )
            cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
                + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
                + cmd
                + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
            output = sp.check_output(cmd.split(' '), universal_newlines=True)
            R_1 = float(output.split('\n')[3].split(' ')[3])
            R_2 = float(output.split('\n')[7].split(' ')[3])
            R_L = float(output.split('\n')[11].split(' ')[3])
            print(output)
        if Print is True:
            rouge_path = join(dec_dir, '../ROUGE.txt')
            with open(rouge_path, 'w') as f:
                print(output, file=f)
        return R_1, R_2, R_L
    
