import pdb, argparse, json
from summarizer import Summarizer

# from get_candidate.py
def load_jsonl(data_path):
    data = []
    model = Summarizer('distilbert-base-uncased')

    out = {}

    with open(data_path) as f:

        #for loop: Here is the canonical code to open a file, 
        # read all the lines out of it, handling one line at a time.

        #sentences = model.sentence_handler(text)

        for idx, line in enumerate(f):
            #json.loads() method can be used 
            # to parse a valid JSON string and convert 
            # it into a Python Dictionary. It is mainly used for
            #  deserializing native string, byte, or byte array 
            # which consists of JSON data into Python Dictionary.
            if(idx > 99):
                break

            #append to data list
            text = json.loads(line)['text']
            result = model(text, num_sentences=5)
            d = {}
            d['sent_id'] = result
            data.append(d)
    return data

parser = argparse.ArgumentParser(
    description='Process truncated documents for indices of important sentences'
)
parser.add_argument('--data_path', type=str, required=True,
    help='path to the original dataset, the original dataset should contain text and summary')
args = parser.parse_args()
data = load_jsonl(args.data_path)

for line in data:
    with open('train_newsroom.jsonl', 'a') as f:
        print(json.dumps(line), file=f)