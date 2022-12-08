import pdb, argparse, json
from summarizer import Summarizer
from summarizer.cluster_features import ClusterFeatures
import numpy as np
from typing import Tuple, List
from time import time
from datetime import timedelta

# from get_candidate.py
def load_jsonl(args):
    data = []
    model = Summarizer('distilbert-base-uncased')

    start = time()

    with open(args.data_path) as f:

        #for loop: Here is the canonical code to open a file, 
        # read all the lines out of it, handling one line at a time.

        #sentences = model.sentence_handler(text)

        for idx, line in enumerate(f):
            #json.loads() method can be used 
            # to parse a valid JSON string and convert 
            # it into a Python Dictionary. It is mainly used for
            #  deserializing native string, byte, or byte array 
            # which consists of JSON data into Python Dictionary.
            if(idx >= args.num_lines):
                break

            #append to data list
            text = json.loads(line)['text']

            # requires editing summary_processor.py in BertExt's summary folder so that cluster_runner returns summary_sentence_indices instead of sentences
            result = model(text, num_sentences=5, min_length=15)

            d = {}
            d['sent_id'] = result
            data.append(d)
            print("Processed line {}".format(idx + 1))

    print('Finished in {}'.format(timedelta(seconds=time()-start)))
    return data

parser = argparse.ArgumentParser(
    description='Process truncated documents for indices of important sentences'
)
parser.add_argument('--data_path', type=str, required=True,
    help='path to the original dataset, the original dataset should contain text and summary')
parser.add_argument('--write_path', type=str, required=True,
    help='path to output file')
parser.add_argument('--num_lines', type=int, required=False,
    help='number of lines to process', default=1000)

args = parser.parse_args()
data = load_jsonl(args)

with open(args.write_path, 'w') as f:
        print('Cleared out file')

for line in data:
    with open(args.write_path, 'a') as f:
        print(json.dumps(line), file=f)
print('Completed writing to {}'.format(args.write_path))