from python_speech_features import logfbank
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
import scipy.io.wavfile as wav
import tensorflow as tf
import numpy as np

n_filters = 26

(rate,sig) = wav.read("test.wav")
fbank_feat = logfbank(sig,rate)
print(fbank_feat.shape)

class Model:

	def __init__(self, hidden_size=256, n_layer=3, n_filters=26, batch_size=5):
		self.hidden_size = hidden_size
		self.n_filters = n_filters
		self.n_layer = n_layer
		self.batch_size = batch_size

		self.input_signal = tf.placeholder(tf.float32, [self.batch_size, None, self.fn_filters], name='input_signal')
		self.output_signal_1 = tf.placeholder(tf.float32, [self.batch_size, None, self.n_filters], name='output_signal1')
		self.output_signal_2 = tf.placeholder(tf.float32, [self.batch_size, None, self.n_filters], name='output_signal2')

	def _net(self):
		rnn_layer = MultiRNNCell([GRUCell(self.hidden_size) for _ in range(self.n_layer)])
		initial_state = rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
		outputs_rnn, state = tf.nn.dynamic_rnn(
			rnn_cell,
			self.input_data,
			initial_state=initial_state,
            dtype=tf.float32)
		input_size = shape(self.input_signal)[2]
		y1_ = tf.layers.dense(inputs=outputs_rnn, units=input_size, activation=tf.nn.relu, name='dense')
		y2_ = tf.layers.dense(inputs=outputs_rnn, units=input_size, activation=tf.nn.relu, name='dense')
		return y1_,y2_

	def loss(self):
		return tf.reduce_mean(tf.square(self.output_signal_1 - y1_)+tf.square(self.output_signal_2 - y2_), name='loss')
