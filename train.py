# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import input_data
import model

EPOCHS = 200
BATCH_SIZE = 20
SHUFFLE_SIZE = 50
VOCAB_SIZE = 284
EMBEDDING_SIZE = 256
LEARNING_RATE = 0.001
SAVE_DIR = "./log/"


if __name__ == "__main__":
    QA_model = model.Seq2Seq(BATCH_SIZE, VOCAB_SIZE, EMBEDDING_SIZE, LEARNING_RATE, True)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tra_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "tra"), sess.graph)
    
    step = 0
    for e in range(EPOCHS):
        for encoder_input_data, decoder_input_data, decoder_input_label in input_data.get_batch("data/text_list", BATCH_SIZE, SHUFFLE_SIZE):
            step += 1
            _, tra_loss, summary_str = sess.run([QA_model.train_op, QA_model.loss, summary_op],
                                                feed_dict={
                                                        QA_model.encoder_input_data: encoder_input_data,
                                                        QA_model.decoder_input_data: decoder_input_data,
                                                        QA_model.decoder_input_label: decoder_input_label,
                                                        QA_model.keep_prob: 0.5
                                                })
            if step % 100 == 0:
                tra_summary_writer.add_summary(summary_str, global_step=step)
                print("Epoch %d, Train Step %d, loss: %.4f" % (e, step, tra_loss))
        saver.save(sess, os.path.join(SAVE_DIR, "model.ckpt"), global_step=step)
    sess.close()

