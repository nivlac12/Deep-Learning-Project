import tensorflow as tf
import pdb
from transformers.modeling_tf_distilbert import TFDistilBertMainLayer
from transformers.modeling_tf_bert import TFBertMainLayer
from transformers import DistilBertConfig, BertConfig
import sys
from metrics import ValidMetric


class MatchSum(tf.keras.Model):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        # self.val_met = tf.metrics.Mean(name="ValidMetric")
        
        if encoder == 'distilbert':
            config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFDistilBertMainLayer(config, trainable=True)
        else:
            config = BertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFBertMainLayer(config, trainable=True)
    
    # def compile(self, val_metric, *args, **kwargs):
    #     super().compile(*args, **kwargs)
        # print(val_metric)
        # self.met = val_metric

    def train_step(self, X): return self.batch_step(X, training=True)
    def test_step(self, X): return self.batch_step(X, training=False)
 
    
    def get_best_candidates(self, scores, candidate_summaries):
        m_idx = tf.cast(tf.argmax(scores, axis = 0), dtype=tf.int32)
        # batch, 1 the value 0-19
        # Issue place
        best_candidates = tf.gather(candidate_summaries, m_idx)
        
        return best_candidates
        
        # tf.map_fn(lambda x: , indices)
        # tf.gather(indices, )
        # indices[:][]
        # tf.gather(, )

    def batch_step(self, X, training):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x1, x2, x3, candidate_summaries, golden_summaries = X[0]
        X = [x1, x2, x3]
        
        with tf.GradientTape() as tape:
            score, summary_score = self(X, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(score, summary_score, regularization_losses=self.losses)
        # diction = {'loss': loss}
        if training:
            # Compute gradients
            diction = {'loss': loss}
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        else:
            best_cands = self.get_best_candidates(score, candidate_summaries)
            self.compiled_metrics.update_state(golden_summaries, best_cands)
            diction = {m.name: m.result() for m in self.metrics}
            diction['loss'] = loss
            self.compiled_metrics.reset()

        return diction
    
    @tf.function
    def call(self, X):
        text_id, candidate_id, summary_id = X
        batch_size = tf.shape(text_id)[0]
        # text_id = [1, 333]
        # candidate_id = [1, 20, 91]
        # summary_id = [1, 33]
        pad_id = 0

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
        summary_score = tf.keras.losses.cosine_similarity(summary_emb, doc_emb, axis=-1)

        # get candidate embedding
        candidate_num = candidate_id.shape[1]
        candidate_id = tf.reshape(candidate_id, shape = (-1, candidate_id.shape[-1]))
        input_mask = ~(candidate_id == pad_id)
        candidate_emb = self.encoder(candidate_id, attention_mask=input_mask)[0]
        # View is a pytorch tensor method. Could be changed with transpose?

        candidate_emb = tf.reshape(candidate_emb[:,0,:], shape = (batch_size, candidate_num, self.hidden_size))
        #assert candidate_emb.shape == (batch_size, candidate_num, self.hidden_size)
        
        temp = tf.shape(tf.transpose(candidate_emb, [1,0,2]))
        doc_emb = tf.broadcast_to(doc_emb, temp)
        doc_emb = tf.transpose(doc_emb, [1,0,2])
        
        score = tf.keras.losses.cosine_similarity(candidate_emb, doc_emb, axis=-1) # [batch_size, candidate_num]
        # assert score.shape == (batch_size, candidate_num)

        return score, summary_score