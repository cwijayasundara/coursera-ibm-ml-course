import numpy as np
import tensorflow as tf
sess = tf.compat.v1.Session()

LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
state

sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.global_variables_initializer())
print (sess.run(state_new))

print (sess.run(output))