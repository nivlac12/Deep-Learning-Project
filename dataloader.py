from time import time
from datetime import timedelta
import json
import pdb
import tensorflow as tf

class MatchSumPipe():

    def __init__(self, max_len=100):
        super(MatchSumPipe, self).__init__()

        self.sep_id = [102] # '[SEP]' (BERT)

        self.max_len = max_len
        
    def process_from_file(self, path):

        print('Start loading datasets !!!')
        start = time()
        cand_dataset, text_dataset, summary_dataset = self._load(path)
        print('Finished in {}'.format(timedelta(seconds=time()-start)))

        return cand_dataset, text_dataset, summary_dataset

    def _load(self, path):
        cand_id, text_id, summ_id = [], [], []

        def quickpad(arr, max):
            paddings = [[0, max-len(arr)]]
            return tf.pad(arr, paddings, 'CONSTANT')

        with open(path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if line_idx > 100:
                    break
                line = json.loads(line)

                cand_id_line = line['candidate_id']
                # skip entries that don't line up
                if len(cand_id_line) != 20:
                    continue
                
                # Truncate candidate id entries if necessary
                new_cand_id = []
                for id_mem in cand_id_line:
                    if len(id_mem) > self.max_len:
                        new_cand_id.append(id_mem[:(self.max_len - 1)] + self.sep_id)
                    else:
                        # pad candidate ids to max len if less than max len
                        cand_line = quickpad(id_mem, self.max_len)
                        new_cand_id.append(cand_line)
                cand_id.append(new_cand_id)

                # pad text line and add
                text_line = quickpad(line['text_id'], 512)
                text_id.append(text_line)

                # pad summary line and add
                summ_line = quickpad(line['summary_id'], 512)
                summ_id.append(summ_line)
        
        cand_id = tf.convert_to_tensor(cand_id)
        text_id = tf.convert_to_tensor(text_id)
        summ_id = tf.convert_to_tensor(summ_id)

        return (cand_id, text_id, summ_id)