import torch

#building blocks for pytorch graphs (so layers mostly)
#https://pytorch.org/docs/stable/nn.html
from torch import nn
from torch.nn import init

from transformers import BertModel, RobertaModel, DistilBertModel

class MatchSum(nn.Module):
    
    #takes in str indicating encoder type (bert or roberta)
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        #storing as self vars so that can access in other methods
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        if encoder == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        elif encoder == 'distilbert':
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.encoder = RobertaModel.from_pretrained('roberta-base')

    #from context, I am gather that text_id is document, candidate_id is
    #the candidate summary, and summary_id is the gold summary, altho will
    #need to confirm
    def forward(self, text_id, candidate_id, summary_id):
        #appears that the 0th dim of the text_id gives the num of batches
        batch_size = text_id.size(0)
        
        #I think this is saying you only need padding if you are using
        #RoBERTa?
        pad_id = 0     # for BERT
        if text_id[0][0] == 0:
            pad_id = 1 # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)

        #puts text id and attention mask (so that not include padding) into 
        #whatever bert encoder was selected by init settings

        #running berta/roberta on document
        out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        #whetever the dim's are of out, 0 seems to be how to pull out the embedding comp
        doc_emb = out[:, 0, :]
        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding

        #~ inverts all bits
        #gets mask so that region inside pad is 0's
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]

        #assertion: will throw error if don't have result you want
        assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        #Returns cosine similarity between x1 and x2, computed along dim. x1 and x2 
        # #must be broadcastable to a common shape. dim refers to the dimension in this common shape. 

        #get cosine similary bw gold sum + doc, I think
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        #2nd dim of candidate_id seems to be num of candidate sums provided by the model?
        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))

        #again, masking candidate summary to avoid the padding and running bert/roberta
        input_mask = ~(candidate_id == pad_id)
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        assert candidate_emb.size() == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1) # [batch_size, candidate_num]
        assert score.size() == (batch_size, candidate_num)

        return {'score': score, 'summary_score': summary_score}

