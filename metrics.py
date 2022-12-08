import numpy as np

import json
from os.path import join
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

#from fastNLP.core.metrics import MetricBase

_ROUGE_PATH = '/path/to/RELEASE-1.5.5'



class MarginRankingLoss(tf.keras.losses.Loss):

    def __init__(self, margin=0.0):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin
    
    @tf.function
    def call(self, score, summary_score):
        y = tf.ones(tf.shape(score))
        TotalLoss = tf.math.reduce_mean(tf.maximum(0.0,-y*(score-score)+0))

        # candidate loss
        n = tf.shape(score)[1]
        for i in range(1, n):
            #includes all but i last vals in seq
            # pos_score = tf.reshape(score[:, :-i], -1)
            pos_score = score[:, :-i]
            #includes only first i values in 2nd dim of seq
            # neg_score = tf.reshape(score[:, i:], -1)
            neg_score = score[:, i:]

            y = tf.ones(tf.shape(pos_score))
            TotalLoss += tf.math.reduce_mean(tf.maximum(0.0,-y*(pos_score-neg_score) + self.margin * tf.cast(i, dtype=tf.float32)))

        # gold summary loss
        # pos_score = tf.reshape(tf.expand_dims(summary_score,-1).broadcast_to(score.shape), -1)
        # neg_score = tf.reshape(score, -1)
        pos_score = tf.broadcast_to(tf.expand_dims(summary_score,-1), tf.shape(score))
        neg_score = score
        y = tf.ones(tf.shape(pos_score))
        TotalLoss += tf.math.reduce_mean(tf.maximum(0.0,-y*(pos_score - neg_score)+0.0))
        return TotalLoss

class ValidMetric():
    def __init__(self, save_path, data, batch_size, max_idx):
        # self.save_path = save_path
        self.data = data

        self.top1_correct = 0
        self.top6_correct = 0
        self.top10_correct = 0
         
        #create instance of Rouge from imported pkg
        self.rouge = Rouge()
        self.ROUGE_val = 0.0
        self.Error = 0
        #I think cur_idx is the index of the current summary
        self.cur_idx = 0
        self.batch_size = batch_size
        self.max_idx = max_idx
    
        self.This_shit_better_work = 0

    def This_shit(self, X):
        self.This_shit_better_work += 1
        return 0
    
    # an approximate method of calculating ROUGE
    def fast_rouge(self, dec, ref):
        if dec == '' or ref == '':
            return 0.0
        
        #get rouge scoesr using dec and ref
        scores = self.rouge.get_scores(dec, ref)

        #seems to return rouge-1*2 + rouge-2/3 but I'm not sure
        return (scores[0]['rouge-1']['f'] + scores[0]['rouge-2']['f'] + scores[0]['rouge-l']['f']) / 3

    def evaluate(self, score):
        # batch_size is just the size of the score vector, which should give you
        # the number of candidate
        # batch_size = tf.shape(score)[0]

        self.top1_correct += tf.math.reduce_sum(tf.cast(tf.math.argmax(score, axis=1) == 0, dtype=tf.int32))
        self.top6_correct += tf.math.reduce_sum(tf.cast(tf.math.argmax(score, axis=1) <= 5, dtype=tf.int32))
        self.top10_correct += tf.math.reduce_sum(tf.cast(tf.math.argmax(score, axis=1) <= 9, dtype=tf.int32))

        # Fast ROUGE  
        # for i in range(self.batch_size):
        for i in range(self.max_idx):
            m_idx = tf.cast(tf.argmax(score[i], axis = 0), dtype=tf.int32)
            # print(self.This_shit_better_work)
            tf.map_fn(lambda x: self.This_shit(x), tf.range(m_idx))
            # print(self.This_shit_better_work)
            # q = 0
            # for _ in tf.range(m_idx):
            #     q+=1
            # m_idx = q
            #replaced max_idx with 20 because the max num of sentences is always 20
            # ext_idx = self.data[self.cur_idx]['indices'][self.This_shit_better_work]
            ext_idx = self.data[i]['indices'][self.This_shit_better_work]

            # print(This_shit_better_work)
            self.This_shit_better_work = 0
            ext_idx.sort()
            dec = []
            # ref = ' '.join(self.data[self.cur_idx]['summary'])
            ref = ' '.join(self.data[i]['summary'])
            for j in ext_idx:
                # dec.append(self.data[self.cur_idx]['text'][j])
                dec.append(self.data[i]['text'][j])
            dec = ' '.join(dec)
            # print(dec)
            # print(ref)
            self.ROUGE_val += self.fast_rouge(dec, ref)
            # self.cur_idx += 1
            # if self.cur_idx == self.max_idx:
            #     break

    def result(self, reset=True): #reset=True):
        # top1_accuracy = self.top1_correct / self.cur_idx
        # top6_accuracy = self.top6_correct / self.cur_idx
        # top10_accuracy = self.top10_correct / self.cur_idx
        top1_accuracy = self.top1_correct / self.max_idx
        top6_accuracy = self.top6_correct / self.max_idx
        top10_accuracy = self.top10_correct / self.max_idx
        ROUGE_ave = self.ROUGE_val / self.max_idx
        eval_result = {'top1_accuracy': top1_accuracy, 'top6_accuracy': top6_accuracy, 
                      'top10_accuracy': top10_accuracy, 'Error': self.Error, 'ROUGE': ROUGE_ave}
        # with open(join(self.save_path, 'train_info.txt'), 'a') as f:
        #     print('top1_accuracy = {}, top6_accuracy = {}, top10_accuracy = {}, Error = {}, ROUGE = {}'.format(
        #           top1_accuracy, top6_accuracy, top10_accuracy, self.Error, ROUGE), file=f)
        # if self.cur_idx == self.max_idx:
        if reset:
            self.top1_correct = 0
            self.top6_correct = 0
            self.top10_correct = 0
            self.ROUGE_val = 0.0
            self.Error = 0
            # self.cur_idx = 0
            # self.This_shit_better_work = 0
        #re-write eval_result so that it is simpler
        eval_result = ROUGE_ave
        return eval_result
        
class MatchRougeMetric():
    def __init__(self, data, dec_path, ref_path, n_total, score=None):
        #super(MatchRougeMetric, self).__init__()
        #self._init_param_map(score=score)
        self.data        = data
        self.dec_path    = dec_path
        self.ref_path    = ref_path
        self.n_total     = n_total
        self.cur_idx = 0
        self.ext = []
        self.start = time()

    
    def evaluate(self, score):
        ext = tf.math.argmax(score, axis=1) # batch_size = 1
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
