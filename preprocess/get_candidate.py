import os
import pdb
import argparse
from os.path import exists
import subprocess as sp
import json
import tempfile
import multiprocessing as mp
from time import time
from datetime import timedelta
import queue
import logging
from itertools import combinations

from cytoolz import curry

#pyrouge doesn't require pytorch
#https://pypi.org/project/pyrouge/
from pyrouge.utils import log

#pyrouge is a Python wrapper for the ROUGE summarization evaluation package. 
#documentation: https://pypi.org/project/pyrouge/
from pyrouge import Rouge155

from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer

MAX_LEN = 512

_ROUGE_PATH = 'C:\\Users\\nivla\\Documents\\CSCI_1470\\pyrouge-master\\tools\\ROUGE-1.5.5'

#dot pulls on parent dir
temp_path = 'temp' # path to store some temporary files

#initialize lists
original_data, sent_ids = [], []

def load_jsonl(data_path):
    data = []
    with open(data_path) as f:

        #for loop: Here is the canonical code to open a file, 
        # read all the lines out of it, handling one line at a time.


        for idx, line in enumerate(f):
            #json.loads() method can be used 
            # to parse a valid JSON string and convert 
            # it into a Python Dictionary. It is mainly used for
            #  deserializing native string, byte, or byte array 
            # which consists of JSON data into Python Dictionary.

            #append to data list
            if(idx > 99):
                break
            data.append(json.loads(line))
    return data

def join(x, y):
    return os.path.join(x,y)

def get_rouge(path, dec):

    #Sets the threshold for this logger to level. Logging 
    # messages which are less severe than level will be ignored; 
    # logging messages which have severity level or higher will be 
    # emitted by whichever handler or handlers service this logger, 
    # unless a handler’s level has been set to a higher severity 
    # level than level.

    #logging.warning: Logs a message with level WARNING on the root 
    # logger. The arguments are interpreted as for debug().

    #log seems to be from: pyrouge.utils.log.get_global_console_logger()
    log.get_global_console_logger().setLevel(logging.WARNING)
    dec_pattern = '(\d+).dec'
    ref_pattern = '#ID#.ref'

    #join: joins 2 items into 1 string. I assume that "path"
    #has / at the end such that this creates two new folders
    dec_dir = join(path, 'decode')
    ref_dir = join(path, 'reference')

    with open(join(dec_dir, '0.dec'), 'w') as f:
        for sentence in dec:
            print(sentence, file=f)

    cmd = '-c 95 -r 1000 -n 2 -m'
    with tempfile.TemporaryDirectory() as tmp_dir:
        #documentation: https://pypi.org/project/pyrouge/
        #To convert plain text summaries into a format ROUGE understands, do:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))

        #To generate the configuration file that ROUGE uses to match system and model summaries, do:
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id=1
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
            + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
            + cmd
            + ' -a {}'.format(join(tmp_dir, 'settings.xml')))

        output = sp.check_output(['perl'] + cmd.split(' '), universal_newlines=True)

        line = output.split('\n')
        rouge1 = float(line[3].split(' ')[3])
        rouge2 = float(line[7].split(' ')[3])
        rougel = float(line[11].split(' ')[3])
    return (rouge1 + rouge2 + rougel) / 3

@curry
def get_candidates(tokenizer, cls, sep_id, idx):

    idx_path = join(temp_path, str(idx))

    #subprocess documentation: https://docs.python.org/3/library/subprocess.html

    #Run the command described by args. Wait for command to complete, 
    # then return the returncode attribute.
    
    # create some temporary files to calculate ROUGE
    sp.call('mkdir ' + idx_path, shell=True)
    sp.call('mkdir ' + join(idx_path, 'decode'), shell=True)
    sp.call('mkdir ' + join(idx_path, 'reference'), shell=True)
    
    # load data
    data = {}
    
    print("idx is {}".format(idx))
    print("text len is {}".format(len(original_data)))
    data['text'] = original_data[idx]['text']
    data['summary'] = original_data[idx]['summary']
    
    # write reference summary to temporary files
    ref_dir = join(idx_path, 'reference')
    with open(join(ref_dir, '0.ref'), 'w') as f:
        for sentence in data['summary']:
            print(sentence, file=f)

    # get candidate summaries
    # here is for CNN/DM: truncate each document into the 5 most important sentences (using BertExt), 
    # then select any 2 or 3 sentences to form a candidate summary, so there are C(5,2)+C(5,3)=20 candidate summaries.
    # if you want to process other datasets, you may need to adjust these numbers according to specific situation.
    sent_id = sent_ids[idx]['sent_id'][:5]

    #combinations arguments: p, r
    #combinations outputs: r-length tuples, in sorted order, no repeated elements
    #seems to return subsequences fo the sent_id sequence of size two
    #and then adds to that list subsequences of sent_id of size 3
    indices = list(combinations(sent_id, 2))
    indices += list(combinations(sent_id, 3))
    if len(sent_id) < 2:
        indices = [sent_id]
    
    # get ROUGE score for each candidate summary and sort them in descending order
    score = []

    #note: i us a str indicating a num
    for i in indices:
        #goes through ea subsequence from the sent_id sequence
        i = list(i)

        #sorts in descending order
        i.sort()
        # write dec
        dec = []
        for j in i:
            #pulls sentence from orig document at the index j
            sent = data['text'][j]
            dec.append(sent)
        try:
            score.append((i, get_rouge(idx_path, dec)))
        except UnicodeDecodeError:
            print('Decode Error')
            continue
    score.sort(key=lambda x : x[1], reverse=True)
    
    # write candidate indices and score
    data['ext_idx'] = sent_id
    data['indices'] = []
    data['score'] = []
    for i, R in score:
        #converts i to int
        data['indices'].append(list(map(int, i)))
        data['score'].append(R)

    # tokenize and get candidate_id
    candidate_summary = []
    for i in data['indices']:
        cur_summary = [cls]
        for j in i:
            cur_summary += data['text'][j].split()
        cur_summary = cur_summary[:MAX_LEN]
        cur_summary = ' '.join(cur_summary)
        candidate_summary.append(cur_summary)
    
    data['candidate_id'] = []
    for summary in candidate_summary:
        token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
        token_ids += sep_id
        data['candidate_id'].append(token_ids)
    
    # tokenize and get text_id
    text = [cls]
    for sent in data['text']:
        text += sent.split()
    text = text[:MAX_LEN]
    text = ' '.join(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['text_id'] = token_ids
    
    # tokenize and get summary_id
    summary = [cls]
    for sent in data['summary']:
        summary += sent.split()
    summary = summary[:MAX_LEN]
    summary = ' '.join(summary)
    token_ids = tokenizer.encode(summary, add_special_tokens=False)[:(MAX_LEN - 1)]
    token_ids += sep_id
    data['summary_id'] = token_ids
    
    # write processed data to temporary file
    processed_path = join(temp_path, 'processed')
    with open(join(processed_path, '{}.json'.format(idx)), 'w') as f:
        json.dump(data, f, indent=4) 
    
    shutil.rmtree(idx_path)

def get_candidates_mp(args):
    
    # choose tokenizer
    #Specified when the Arg parseris used to run the file
    if args.tokenizer == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        cls, sep = '[CLS]', '[SEP]'
    elif args.tokenizer == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        cls, sep = '<s>', '</s>'
    else:
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        cls, sep = '[CLS]', '[SEP]'
    sep_id = tokenizer.encode(sep, add_special_tokens=False)

    # load original data and indices
    global original_data, sent_ids

    #loads in list consistenting of ea line of the jsonl file in the
    #data path
    #seems to do this for original documents that have been preprocessed
    #by BERTExt
    #then repeats this pulling from the jsonl file only containing
    #the sentences deemed by BERTExt to be relevant
    #Pulls from the path specified by theargs parser when the file is run
    original_data = load_jsonl(args.data_path)
    sent_ids = load_jsonl(args.index_path)

    #gets the total num of sentences in the 
    n_files = len(original_data)

    #Assertion to ensure that the indices given for the sentences that were pulled out of the original document
    #Is equal to the number of sentences that were pulled out of the original data
    assert len(sent_ids) == len(original_data)
    print('total {} documents'.format(n_files))
    os.makedirs(temp_path)
    processed_path = join(temp_path, 'processed')
    os.makedirs(processed_path)

    # use multi-processing to get candidate summaries
    start = time()
    print('start getting candidates with multi-processing !!!')


    for i in range(100):
        get_candidates(tokenizer, cls, sep_id, i)

    '''
    with mp.Pool() as pool:
        pdb.set_trace()
        list(pool.imap_unordered(get_candidates(tokenizer, cls, sep_id), range(n_files), chunksize=20))
    '''
    
    
    print('finished in {}'.format(timedelta(seconds=time()-start)))
    
    # write processed data
    print('start writing {} files'.format(n_files))
    for i in range(n_files):
        with open(join(processed_path, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        with open(args.write_path, 'a') as f:
            print(json.dumps(data), file=f)
    
    os.system('rm -r {}'.format(temp_path))

if __name__ == '__main__':
    
    #A command-line interpreter or command-line processor uses a command-line interface 
    # to receive commands from a user in the form of lines of text. This 
    # provides a means of setting parameters for the environment, invoking 
    # executables and providing information to them as to what actions they 
    # are to perform

    #create instance of argparser
    parser = argparse.ArgumentParser(
        description='Process truncated documents to obtain candidate summaries'
    )

    # remove temp directory
    import shutil
    shutil.rmtree('temp')

    #Takes in arguments input into theparserwhen the script is run
    parser.add_argument('--tokenizer', type=str, required=True,
        help='BERT/RoBERTa')
    parser.add_argument('--data_path', type=str, required=True,
        help='path to the original dataset, the original dataset should contain text and summary')
    parser.add_argument('--index_path', type=str, required=True,
        help='indices of the remaining sentences of the truncated document')
    parser.add_argument('--write_path', type=str, required=True,
        help='path to store the processed dataset')

    #The ArgumentParser.parse_args() method runs the parser and places the 
    # extracted data in a argparse.Namespace object:
    args = parser.parse_args()

    #Assertion lines to ensure thatthe tokenizer selected is either burnt or Roberta
    #Assertion to check that the data path specified Exist
    assert args.tokenizer in ['bert', 'roberta']
    assert exists(args.data_path)
    assert exists(args.index_path)

    #Puts parser arguments into thefunction below
    get_candidates_mp(args)
