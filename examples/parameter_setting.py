#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-06-30
Brief:  Parameter setting of runing models
"""

import copy
from collections import OrderedDict


data_path = "../data/semeval_mic_train_and_test_no_extraction"

# Parameters
p_fnn = OrderedDict([
    ("\nParameters for word vectors", ""),
    # ("word2vec_path", "../data/sample_word2vec.txt"),
    ("word2vec_path", "../../word2vec/vector_model/glove.6B.300d.txt"),
    # ("word2vec_path", "../../word2vec/vector_model/glove.840B.300d.txt"),
    ("norm_vec", False),
    ("oov", "O_O_V"),
    ("\nParameters for loading data", ""),
    ("data_path",data_path),
    ("left_win", 5),
    ("right_win", 5),
    ("use_verb", True),
    ("lower", True),
    # Validation part and train_part are from train_data_path
    ("train_part", 0.7),
    ("validation_part", 0.1),
    ("test_part", 0.2),
    # Minimum number of sentences of training data
    ("minimum_sent_num", 70),
    # Minimum frame of verb of training data
    ("minimum_frame", 2),
    ("\nParameters for NN model", ""),
    ("n_hs", []),
    ("up_wordvec", False),
    ("use_bias", True),
    ("act_func", "tanh"),
    ("max_epochs", 100),
    ("minibatch", 5),
    ("lr", 0.1),
    ("random_vectors", False),
    ("\nOther parameters", ""),
    ("prediction_results",
     "../result/new_fnn_results/nothing")
])

p_fnn_logic_test = copy.copy(p_fnn)
p_fnn_logic_test["prediction_results"] = "../result/new_fnn_results/logic_test"
p_fnn_logic_test["word2vec_path"] = "../data/sample_word2vec.txt"


p_rnn = OrderedDict([
    ("\nParameters for word vectors", ""),
    # ("word2vec_path", "../data/sample_word2vec.txt"),
    ("word2vec_path", "../../word2vec/vector_model/glove.6B.300d.txt"),
    # ("word2vec_path", "../../word2vec/vector_model/glove.840B.300d.txt"),
    ("norm_vec", False),
    ("oov", "O_O_V"),
    ("\nParameters for loading data", ""),
    ("data_path",data_path),
    ("left_win", -1),
    ("right_win", -1),
    ("use_verb", True),
    ("lower", True),
    ("use_padding", False),
    # Validation part and train_part are from train_data_path
    ("train_part", 0.7),
    ("validation_part", 0.1),
    ("test_part", 0.2),
    # Minimum number of sentences of training data
    ("minimum_sent_num", 70),
    # Minimum frame of verb of training data
    ("minimum_frame", 2),
    ("\nParameters for rnn model", ""),
    ("n_h", 30),
    ("up_wordvec", False),
    ("use_bias", True),
    ("act_func", "tanh"),
    ("use_lstm", False),
    ("max_epochs", 100),
    ("minibatch", 5),
    ("lr", 0.01),
    ("random_vectors", False),
    ("\nOther parameters", ""),
    ("prediction_results",
     "../result/rnn_results/490train_nopadding_lr0.01_rnn_win11")
])

p_rnn_logic_test = copy.copy(p_rnn)
p_rnn_logic_test["prediction_results"] = "../result/rnn_results/logic_test"
p_rnn_logic_test["word2vec_path"] = "../data/sample_word2vec.txt"

p_lstm = copy.copy(p_rnn)
p_lstm["use_lstm"] = True

p_lstm_logic_test = copy.copy(p_rnn_logic_test)
p_lstm_logic_test["use_lstm"] = True


p_birnn = OrderedDict([
    ("\nParameters for word vectors", ""),
    # ("word2vec_path", "../data/sample_word2vec.txt"),
    ("word2vec_path", "../../word2vec/vector_model/glove.6B.300d.txt"),
    # ("word2vec_path", "../../word2vec/vector_model/glove.840B.300d.txt"),
    ("norm_vec", False),
    ("oov", "O_O_V"),
    ("\nParameters for loading data", ""),
    ("data_path",data_path),
    ("left_win", -1),
    ("right_win", -1),
    ("use_verb", True),
    ("lower", True),
    ("use_padding", False),
    ("verb_index", True),
    # Validation part and train_part are from train_data_path
    ("train_part", 0.7),
    ("validation_part", 0.1),
    ("test_part", 0.2),
    # Minimum number of sentences of training data
    ("minimum_sent_num", 70),
    # Minimum frame of verb of training data
    ("minimum_frame", 2),
    ("\nParameters for rnn model", ""),
    ("n_h", 30),
    ("up_wordvec", False),
    ("use_bias", True),
    ("act_func", "tanh"),
    ("use_lstm", False),
    ("max_epochs", 100),
    ("minibatch", 5),
    ("lr", 0.1),
    ("random_vectors", False),
    ("\nOther parameters", ""),
    ("prediction_results",
     "../result/brnn_results/70train_blstm_0.1lr_nopadding_win11")
])

p_birnn_logic_test = copy.copy(p_birnn)
p_birnn_logic_test["prediction_results"] = "../result/brnn_results/logic_test"
p_birnn_logic_test["word2vec_path"] = "../data/sample_word2vec.txt"

p_bilstm = copy.copy(p_birnn)
p_bilstm["use_lstm"] = True

p_bilstm_logic_test = copy.copy(p_birnn_logic_test)
p_bilstm_logic_test["use_lstm"] = True
