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
