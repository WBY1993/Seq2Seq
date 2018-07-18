# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
import input_data


class Seq2Seq:
    def __init__(self, batch_size, vocab_size, embedding_size, lr, is_train):
        # initial
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lr = lr
        self.is_train = is_train
        
        # input_data
        self.encoder_input_data, self.decoder_input_data, self.decoder_input_label, self.input_seq_len, self.output_seq_len, self.keep_prob = self.get_input()
        
        # build graph
        self.logits, self.pred = self.inference()
        self.loss = self.losses()
        self.train_op = self.training()
        
    def get_input(self):
        encoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="encoder_input_data")
        input_seq_len = tf.reduce_sum(tf.sign(encoder_input_data), reduction_indices=1)
        decoder_input_data = tf.placeholder(tf.int32, shape=[None, None], name="decoder_input_data")
        decoder_input_label = tf.placeholder(tf.int32, shape=[None, None], name="decoder_inout_label")
        output_seq_len = tf.reduce_sum(tf.sign(decoder_input_data), reduction_indices=1)
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        return encoder_input_data, decoder_input_data, decoder_input_label, input_seq_len, output_seq_len, keep_prob
    
    def inference(self):
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            encoder_input_data_embedding = tf.nn.embedding_lookup(embedding, self.encoder_input_data)
            decoder_input_data_embedding = tf.nn.embedding_lookup(embedding, self.decoder_input_data)
        
        with tf.variable_scope("encoder"):
            en_lstm1 = rnn.BasicLSTMCell(256)
            en_lstm1 = rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
            en_lstm2 = rnn.BasicLSTMCell(256)
            en_lstm2 = rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
            encoder_cell_fw = rnn.MultiRNNCell([en_lstm1])
            encoder_cell_bw = rnn.MultiRNNCell([en_lstm2])
        bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(encoder_cell_fw,
                                                                               encoder_cell_bw,
                                                                               encoder_input_data_embedding,
                                                                               sequence_length=self.input_seq_len,
                                                                               dtype=tf.float32)
        encoder_outputs = tf.concat(bi_encoder_outputs, -1)
        encoder_state = []
        for layer_id in range(1):  # layer_num
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
        encoder_state = tuple(encoder_state)
        
        with tf.variable_scope("decoder"):
            de_lstm1 = rnn.BasicLSTMCell(256)
            de_lstm1 = rnn.DropoutWrapper(de_lstm1, output_keep_prob=self.keep_prob)
            de_lstm2 = rnn.BasicLSTMCell(256)
            de_lstm2 = rnn.DropoutWrapper(de_lstm2, output_keep_prob=self.keep_prob)
            decoder_cell = rnn.MultiRNNCell([de_lstm1, de_lstm2])
            
            attention_mechanism = seq2seq.LuongAttention(256, encoder_outputs, self.input_seq_len)
            decoder_cell = seq2seq.AttentionWrapper(decoder_cell, attention_mechanism, 256)
            decoder_initial_state = decoder_cell.zero_state(self.batch_size, dtype=tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
            
            output_projection = Dense(self.vocab_size, name="output_projection")
            if self.is_train:
                helper = seq2seq.TrainingHelper(decoder_input_data_embedding, self.output_seq_len)
                decoder = seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=output_projection)
                decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder)
                logits = decoder_outputs.rnn_output
                pred = decoder_outputs.sample_id
            else:
                # #################SampleEmbedding#################
                helper = seq2seq.SampleEmbeddingHelper(embedding,
                                                       start_tokens=[input_data.GO_ID] * self.batch_size,
                                                       end_token=input_data.EOS_ID)
                # #################GreedyEmbedding#################
                # helper = seq2seq.GreedyEmbeddingHelper(embedding,
                #                                        start_tokens=[input_data.GO_ID] * self.batch_size,
                #                                        end_token=input_data.EOS_ID)
                decoder = seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state, output_layer=output_projection)
                decoder_outputs, _, _ = seq2seq.dynamic_decode(decoder, maximum_iterations=10)
                logits = decoder_outputs.rnn_output
                pred = decoder_outputs.sample_id
            return logits, pred
        
    def losses(self):
        with tf.variable_scope("loss"):
            weights = tf.sequence_mask(tf.to_int32(self.output_seq_len), tf.to_int32(tf.shape(self.decoder_input_data)[1]))
            loss = seq2seq.sequence_loss(self.logits, self.decoder_input_label, tf.to_float(weights))
            tf.summary.scalar("loss", loss)
        return loss

    def training(self):
        with tf.variable_scope("optimizer"):
            decayed_lr = tf.train.exponential_decay(learning_rate=self.lr,
                                                    global_step=self.global_step,
                                                    decay_steps=5000,
                                                    decay_rate=0.5,
                                                    staircase=True)
            optimizer = tf.train.AdamOptimizer(decayed_lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 5)
            grads_vars = zip(grads, tvars)
            train_op = optimizer.apply_gradients(grads_vars, self.global_step)
            return train_op
