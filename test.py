# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import jieba
import sys
import input_data
import model

BATCH_SIZE = 2
VOCAB_SIZE = 284
EMBEDDING_SIZE = 256
LEARNING_RATE = 0.01
LOG_DIR = "./log/"


if __name__ == "__main__":
    QA_model = model.Seq2Seq(BATCH_SIZE, VOCAB_SIZE, EMBEDDING_SIZE, LEARNING_RATE, False)
    saver = tf.train.Saver()
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found')
    
    with open("data/vocab", "r", encoding="utf-8") as file:
        vocab = file.read()
        vocab = eval(vocab)
        vocab2id = vocab
        id2vocab = {v:k for k,v in vocab.items()}
    
    while True:
        question = sys.stdin.readline()
        question = question.strip()
        if not question:
            break
        question_id = []
        for word in jieba.cut(question):
            try:
                question_id.append(vocab2id[word])
            except KeyError:
                if word.replace(".", "", 1).isdigit():
                    question_id.append(input_data.NUM_ID)
                else:
                    question_id.append(input_data.UNK_ID)
        
        encoder_input_data = []
        decoder_input_data = []
        encoder_input_data.append(np.array(question_id, dtype=np.int32))
        
        preds = sess.run(QA_model.pred, feed_dict={QA_model.encoder_input_data: encoder_input_data * BATCH_SIZE,
                                                   QA_model.keep_prob: 1})
        
        pred = preds[0, :]
        output = []
        for num in pred:
            if num == 0:
                output.append("PAD_ID")
            elif num == 1:
                output.append("GO_ID")
            elif num == 2:
                break
            elif num == 3:
                output.append("NUM_ID")
            elif num == 4:
                output.append("UNK_ID")
            elif num < 10:
                output.append("UNK_ID")
            else:
                output.append(id2vocab[num])
        print("".join(output))
    
    sess.close()

