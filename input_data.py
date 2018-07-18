# -*- coding: utf-8 -*-
import numpy as np
import jieba
import Pinyin2Hanzi

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
NUM_ID = 3
UNK_ID = 4


def get_vocab_dic(file_list, vocab_size=60000):
    vocab_dict = {}
    for file_name in file_list:
        with open(file_name, "r", encoding="utf-8") as file:
            line_num = 0
            for line in file:
                line_num += 1
                if line_num % 10000 == 0:
                    print("file_name:%s line:%d" % (file_name, line_num))
                line = line.strip()
                for word in jieba.cut(line):
                    if Pinyin2Hanzi.is_chinese(word):
                        if word not in vocab_dict.keys():
                            vocab_dict[word] = 1
                        else:
                            vocab_dict[word] += 1
                    elif word in [" ", ":", ";", "\"", "'", "[", "]", "{", "}", ",", ".", "/", "?", "~", "!", "@", "#", "$",
                                  "%", "^", "&", "*", "(", ")", "-", "_", "+", "="]:
                        if word not in vocab_dict.keys():
                            vocab_dict[word] = 1
                        else:
                            vocab_dict[word] += 1
                    elif word in [" ", "：", "；", "“", "”", "‘", "「", "」", "{", "}", "，", "。", "/", "？", "～", "！", "@", "#", "￥",
                                  "%", "…", "&", "×", "（", "）", "-", "—", "+", "="]:
                        if word not in vocab_dict.keys():
                            vocab_dict[word] = 1
                        else:
                            vocab_dict[word] += 1
    vocab_list = sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True)
    if len(vocab_list) > vocab_size:
        vocab_list = vocab_list[:vocab_size]
    vocab = {}
    with open("data/vocab_list", "w", encoding="utf-8") as file:
        for i,c in enumerate(vocab_list):
            vocab[c[0]] = i + 10
            file.write("{},{}\n".format(i+10, c))
    with open("data/vocab", "w", encoding="utf-8") as file:
        file.write(str(vocab))


def text_converter(question_file, answer_file):
    with open("data/vocab", "r", encoding="utf-8") as file:
        vocab = file.read()
        vocab = eval(vocab)
        vocab2id = vocab

    question_len = []
    answer_len = []
    file_q = open(question_file, "r", encoding="utf-8")
    file_a = open(answer_file, "r", encoding="utf-8")
    file_w = open("data/text_list", "w", encoding="utf-8")
    line_num = 0
    while True:
        line_num += 1
        if line_num % 10000 == 0:
            print(line_num)
        question = file_q.readline()
        answer = file_a.readline()
        if question and answer:
            question = question.strip()
            answer = answer.strip()
            if not question or not answer:
                continue

            question_id = []
            for word in jieba.cut(question):
                try:
                    question_id.append(vocab2id[word])
                except KeyError:
                    if word.replace(".", "", 1).isdigit():
                        question_id.append(NUM_ID)
                    else:
                        question_id.append(UNK_ID)

            answer_id = []
            for word in jieba.cut(answer):
                try:
                    answer_id.append(vocab2id[word])
                except KeyError:
                    if word.replace(".", "", 1).isdigit():
                        answer_id.append(NUM_ID)
                    else:
                        answer_id.append(UNK_ID)

            question_len.append(len(question_id))
            answer_len.append(len(answer_id))
            print([question_id, answer_id], file=file_w)
        else:
            break
    file_q.close()
    file_a.close()
    file_w.close()
    print("Question length: %d--%d" % (min(question_len), max(question_len)))
    print("Answer length: %d--%d" % (min(answer_len), max(answer_len)))


def padding(encoder_input, decoder_input):
    input_max_len = max([len(i) for i in encoder_input])
    output_max_len = max([len(i) for i in decoder_input])
    encoder_input_data = []
    decoder_input_data = []
    decoder_input_label = []
    for i in range(len(encoder_input)):
        encoder_input_data.append(np.array(encoder_input[i] + [PAD_ID] * (input_max_len-len(encoder_input[i])), dtype=np.int32))
        decoder_input_data.append(np.array([GO_ID] + decoder_input[i] + [PAD_ID] * (output_max_len-len(decoder_input[i])), dtype=np.int32))
        decoder_input_label.append(np.array(decoder_input[i] + [EOS_ID] + [PAD_ID] * (output_max_len-len(decoder_input[i])), dtype=np.int32))
    return encoder_input_data, decoder_input_data, decoder_input_label
    
    

def get_batch(text_file, batch_size, shuffle_size):
    file = open(text_file, "r", encoding="utf-8")

    text_list = []
    encoder_input = []
    decoder_input = []


    batch_num = 0
    while True:
        data = file.readline()
        if data:
            data = eval(data)
            text_list.append(data)
            if len(text_list) == shuffle_size:
                np.random.shuffle(text_list)
                for _ in range(batch_size):
                    sample = text_list.pop()
                    encoder_input.append(sample[0])
                    decoder_input.append(sample[1])
                    batch_num += 1
                    if batch_num == batch_size:
                        encoder_input_data, decoder_input_data, decoder_input_label = padding(encoder_input, decoder_input)
                        yield encoder_input_data, decoder_input_data, decoder_input_label
                        encoder_input.clear()
                        decoder_input.clear()
                        batch_num = 0
        else:
            file.close()
            while len(text_list)>0:
                sample = text_list.pop()
                encoder_input.append(sample[0])
                decoder_input.append(sample[1])
                batch_num += 1
                if batch_num == batch_size:
                    encoder_input_data, decoder_input_data, decoder_input_label = padding(encoder_input, decoder_input)
                    yield encoder_input_data, decoder_input_data, decoder_input_label
                    encoder_input.clear()
                    decoder_input.clear()
                    batch_num = 0
            break
