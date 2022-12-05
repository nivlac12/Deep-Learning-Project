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
        self.val_met = tf.metrics.Mean(name="ValidMetric")
        
        if encoder == 'distilbert':
            config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFDistilBertMainLayer(config, trainable=True)
        else:
            config = BertConfig.from_pretrained('distilbert-base-uncased')
            self.encoder = TFBertMainLayer(config, trainable=True)

    def compile(self, val_metric, *args, **kwargs):
        super().compile(*args, **kwargs)
        print(val_metric)
        self.met = val_metric


    def train_step(self, X): return self.batch_step(X, training=True)
    def test_step(self, X): return self.batch_step(X, training=False)

    def batch_step(self, X, training):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X = X[0]

        with tf.GradientTape() as tape:
            score, summary_score = self(X, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(score, summary_score, regularization_losses=self.losses)
        
        self.met.evaluate(score)
        self.val_met.update_state(self.met.result())

        if training:
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        # print(self.compiled_metrics)
        # self.compiled_metrics.update_state(score, tf.ones(tf.shape(score)))
        diction = {'loss': loss}

        diction['valid_metric'] = self.val_met.result()
        # Return a dict mapping metric names to current value
        # diction = {m.name: m.result() for m in self.metrics}
        # print(diction)
        # sys.exit(0)
        # return {m.name: m.result() for m in self.metrics}
        return diction

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