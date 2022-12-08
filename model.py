import tensorflow as tf
import pdb
from transformers.modeling_tf_distilbert import TFDistilBertMainLayer
from transformers.modeling_tf_bert import TFBertMainLayer
from transformers import DistilBertConfig, BertConfig
import sys


class MatchSum(tf.keras.Model):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        self.total_rouge = tf.Variable(0.0, dtype=tf.float32, trainable=False)

        if encoder == 'distilbert':
            config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFDistilBertMainLayer(config, trainable=True)
        else:
            config = BertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFBertMainLayer(config, trainable=True)

    def train_step(self, X): return self.batch_step(X, training=True)
    def test_step(self, X): return self.batch_step(X, training=False)
 
    def get_best_candidates(self, scores, candidate_summaries):
        max_score_idx = tf.argmax(scores, axis=1)
        best_candidates = tf.gather(params=candidate_summaries, indices=max_score_idx, axis=1, batch_dims=1)
        return best_candidates

    def batch_step(self, X, training):
        candidate_summaries, golden_summaries = X[1]    
        
        with tf.GradientTape() as tape:
            score, summary_score = self(X[0], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(score, summary_score, regularization_losses=self.losses)
        diction = {'loss': loss}
        if training:
            # Compute gradients 
            gradients = tape.gradient(loss, self.trainable_variables)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            best_cands = self.get_best_candidates(score, candidate_summaries)
            self.compiled_metrics.update_state(golden_summaries, best_cands)
            for m in self.compiled_metrics._user_metrics:
                self.total_rouge.assign_add(m.result()['f1_score'])
                m.reset_states()
            diction['averaged rouge f1_scores'] = tf.math.divide(self.total_rouge, 3.0)
            self.total_rouge.assign(0.0)

        return diction
    
    @tf.function
    def call(self, X):
        text_id, candidate_id, summary_id = X
        batch_size = tf.shape(text_id)[0]
        pad_id = 0

        # get document embedding
        input_mask = ~(text_id == pad_id)
        out = self.encoder(text_id, attention_mask=input_mask)[0] # last layer
        doc_emb = out[:, 0, :]
        
        # get summary embedding
        input_mask = ~(summary_id == pad_id)
        out = self.encoder(summary_id, attention_mask=input_mask)[0] # last layer
        summary_emb = out[:, 0, :]

        # get summary score
        summary_score = tf.keras.losses.cosine_similarity(summary_emb, doc_emb, axis=-1)

        # get candidate embedding
        candidate_num = candidate_id.shape[1]
        candidate_id = tf.reshape(candidate_id, shape = (-1, candidate_id.shape[-1]))
        input_mask = ~(candidate_id == pad_id)
        candidate_emb = self.encoder(candidate_id, attention_mask=input_mask)[0]

        candidate_emb = tf.reshape(candidate_emb[:,0,:], shape = (batch_size, candidate_num, self.hidden_size))
        
        temp = tf.shape(tf.transpose(candidate_emb, [1,0,2]))
        doc_emb = tf.broadcast_to(doc_emb, temp)
        doc_emb = tf.transpose(doc_emb, [1,0,2])
        
        score = tf.keras.losses.cosine_similarity(candidate_emb, doc_emb, axis=-1) # [batch_size, candidate_num]

        return score, summary_score