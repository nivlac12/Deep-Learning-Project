import os
from os.path import exists, join
import json
import tensorflow as tf

def get_all_summaries(path, num_samples=100):
    json_file = read_jsonl(path, num_samples=num_samples)
    golden_summaries = []
    candidate_summaries = []

    for article in json_file:
        golden_summaries.append(' '.join(article['summary']))

        article_cands = []
        indices = article['indices']
        text = article['text']

        for cand_indices in indices:
            cand_sum = []
            for i in cand_indices:
                cand_sum.append(text[i])
            article_cands.append(' '.join(cand_sum))
        candidate_summaries.append(article_cands)
    

    return tf.convert_to_tensor(golden_summaries), tf.convert_to_tensor(candidate_summaries)

#same fun as in preprocess. seems to turn each line of jsonl
#file to different items fo the same list, in order
def read_jsonl(path, num_samples=100):
    data = []
    with open(path) as f:
        for line_idx, line in enumerate(f):
        # for line in f:
            if line_idx > num_samples:
                break
            line = json.loads(line)

            cand_id_line = line['candidate_id']
            # skip entries that don't line up
            if len(cand_id_line) != 20 or line['score'] == []:
                continue
            data.append(line)
    return data

#returns a dict of paths for the loc of .sonl data files for training,
#validation, testing

#does not seem to require torch pkgs
def get_data_path(mode, encoder):

    #def path as dictionary
    paths = {}

    #for supplying data path into data folder

    #for training: put two different keys into dict, one for train
    #and another for validation
    if mode == 'train':
        paths['train'] = 'data/train_CNNDM_' + encoder + '.jsonl'
        paths['val']   = 'data/val_CNNDM_' + encoder + '.jsonl'
    else:
        paths['test']  = 'data/test_CNNDM_' + encoder + '.jsonl'
    return paths

#seems to, if necessary, create a result dir inside save_path,
#it then creates a directory for the model inside that dir
#if one does not exist 
#then makes dec and ref paths inside that dir if those do not exist
#it then returns the str's of the paths that it just made

#note: requires no tf modules
def get_result_path(save_path, cur_model):

    #join: joins two strings
    result_path = join(save_path, '../result')
    if not exists(result_path):
        os.makedirs(result_path)
    model_path = join(result_path, cur_model)

    #checks if path exists
    if not exists(model_path):
        #Then os.makedirs() method will create all unavailable/missing directory in the specified path. 
        os.makedirs(model_path)
    dec_path = join(model_path, 'dec')
    ref_path = join(model_path, 'ref')

    #Recursive directory creation function. Like mkdir(), but makes 
    # all intermediate-level directories needed to contain the leaf directory.
    os.makedirs(dec_path)
    os.makedirs(ref_path)
    return dec_path, ref_path
