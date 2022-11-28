import torch
import torch.nn as nn
import tensorflow as tf

input_one = torch.randn(3, requires_grad=True)
input_two = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(input_one, input_two, target)
output.backward()

print('input one: ', input_one)
print('input two: ', input_two)
print('target: ', target)
print('output: ', output)

a = tf.constant([[1, 2],
                [3, 4],
                [5, 6]], dtype=tf.float16)
b = tf.constant([[5, 9],
                [3, 6],
                [1, 8]], dtype=tf.float16)
y_pred = tf.linalg.norm(a - b, axis=1)
print(y_pred)
