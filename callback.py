import os
import sys
import tensorflow as tf
from os.path import join
import numpy as np
# from keras import backend as K

class MyCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, max_lr=0.0, warmup_steps = 0, update_every = 0):
        super(MyCallback, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        self.real_step = 0
        # self.lrs = []

#     # def on_valid_begin(self):
#     #     with open(join(self._trainer.save_path, 'train_info.txt'), 'a') as f:
#     #         print('Current step is: {}'.format(self.step), file=f)

    # def on_batch_end(self, batch, logs=None):
    #     self.global_step = self.global_step + 1
    #     lr = self.model.optimizer.lr.numpy()
    #     self.lrs.append(lr)
    
    def on_batch_begin(self, batch, logs=None):
        print(batch)
        if batch % self.update_every == 0 and batch > 0:
            self.real_step += 1
            cur_lr = self.max_lr * 100 * min(self.real_step ** (-0.5), self.real_step * self.warmup_steps**(-1.5))
            # K.set_value(self.model.optimizer.lr, cur_lr)
            tf.compat.v2.keras.backend.set_value(self.model.optimizer.lr, cur_lr)
            # if self.real_step % 1000 == 0:
                # self.pbar.write('Current learning rate is {:.8f}, real_step: {}'.format(cur_lr, self.real_step))
                # print('Current learning rate is {:.8f}, real_step: {}'.format(cur_lr, self.real_step))

