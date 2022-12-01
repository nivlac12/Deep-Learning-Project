import tensorflow as tf
import pdb

from transformers import TFBertModel, TFRobertaModel, TFDistilBertModel

class MatchSum(tf.keras.Model):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        if encoder == 'distilbert':
            self.encoder = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        elif encoder == 'bert':
            self.encoder = TFBertModel.from_pretrained('bert-base-uncased') 
        else:
            self.encoder = TFRobertaModel.from_pretrained('roberta-base')

    def call(self, X):
        text_id, candidate_id, summary_id = X
        batch_size = tf.shape(text_id)[0]
        
        # text_id = [1, 333]
        # candidate_id = [1, 20, 91]
        # summary_id = [1, 33]
        pad_id = 0     # for BERT or DistilBERT?
        if text_id[0][0] == 0:
            pad_id = 1 # for RoBERTa

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        doc_emb = out[:, 0, :]

        #assert doc_emb.shape == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]
        #assert summary_emb.shape == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        # tf.keras.metrics.CosineSimilarity or tf.keras.losses.CosineSimilarity
        summary_score = tf.Variable(tf.keras.losses.cosine_similarity(summary_emb, doc_emb, axis=-1), trainable=True)

        # get candidate embedding
        candidate_num = candidate_id.shape[1]
        # View is a pytorch tensor method. Could be changed with transpose?
        # candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        candidate_id = tf.reshape(candidate_id, shape = (-1, candidate_id.shape[-1]))
        input_mask = ~(candidate_id == pad_id)
        candidate_emb = self.encoder(candidate_id, attention_mask=input_mask)[0]
        # View is a pytorch tensor method. Could be changed with transpose?

        candidate_emb = tf.reshape(candidate_emb[:,0,:], shape = (batch_size, candidate_num, self.hidden_size))
        # candidate_emb = out[:, 0, :].view(batch_size, candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]
        #assert candidate_emb.shape == (batch_size, candidate_num, self.hidden_size)
        
        # get candidate score
        # These are pytorch tensor commands.
        doc_emb = tf.broadcast_to(tf.expand_dims(doc_emb, axis=0), tf.shape(candidate_emb)) # This could be the incorrect axis. Also uncertain about the broadcast to parameter input
        # doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        
        # tf.keras.metrics.CosineSimilarity or tf.keras.losses.CosineSimilarity
        score = tf.Variable(tf.keras.losses.cosine_similarity(candidate_emb, doc_emb, axis=-1), trainable=True) # [batch_size, candidate_num]
        #assert score.shape == (batch_size, candidate_num)

        pdb.set_trace()
        return score, summary_score