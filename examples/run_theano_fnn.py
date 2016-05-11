#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/29
Brief:  Examples of running models
"""

import numpy as np
from collections import OrderedDict
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)
import sys
sys.path.append("../utils/")
sys.path.append("../models/")
from tools import*
from data_loader import DataLoader
from metrics import*
from theano_fnn import FNN

def gen_print_info(field_names, values):
    """
    Generate print infomation
    field_names: 1d array-like, each element of field_names is string
    values: 1d array-like. The value of field_names
    return: string
    """
    if len(field_names) != len(values):
        logging.error("The length is not the same.field_names:%s, values:%s" %\
                (str(field_names), str(values)))
        raise Exception
    res = ""
    for i in range(0, len(field_names)):
        res += "%s:%s\t" % (field_names[i], values[i])
    return res

def run_fnn():
    # Parameters
    p = OrderedDict([
        ("\nParameters for word vectors", ""), 
        ("word2vec_path", "../data/data2vec.txt"), 
        # ("word2vec_path", "../../word2vec/vector_model/glove.6B.300d.txt"), 
        # ("word2vec_path", "../../word2vec/vector_model/glove.840B.300d.txt"), 
        ("norm_vec", False), 
        ("oov", "O_O_V"), 
        ("\nParameters for loading data", ""), 
        # ("train_data_path", "../data/semeval_mic_test_and_pdev_train/train/"), 
        # ("test_data_path", "../data/semeval_mic_test_and_pdev_train/test/"), 
        ("train_data_path", "../data/split_pdev/train/"), 
        ("test_data_path", "../data/split_pdev/test/"), 
        ("left_win", 5), 
        ("right_win", 5), 
        ("use_verb", False), 
        ("lower", True), 
        ("train_part", 0.9), 
        ("validation_part", 0.1),   # Validation part and train_part are from train_data_path
        ("test_part", 1.0), 
        ("minimum_sent_num", 70), # Minimum number of sentences of training data
        ("minimum_frame", 2),  # Minimum frame of verb of training data
        ("\nParameters for FNN model", ""), 
        ("n_hidden", 30), 
        ("weight", 0.0), 
        ("up_wordvec", False), 
        ("max_epochs", 100), 
        ("minibatch", 5), 
        ("lr", 0.1), 
        ("l2_factor", 0.0001), 
        ("early_stopping", True), 
        ("random_vectors", False), 
        ("\nOther parameters", ""), 
        ("prediction_results", "../result/word_embedding_tests/prediction_results401verbs_6Blower_upvectors") 
    ])

    # Get vocabulary and word vectors
    vocab, invocab, word2vec = get_vocab_and_vectors(
        p["word2vec_path"], norm_only=p["norm_vec"], oov=p["oov"],
        oov_vec_padding=0., dtype=FLOAT, file_format="auto"
    )
    if p["random_vectors"]:
        word2vec = np.array(np.random.uniform(low=-0.5, high=0.5, size=word2vec.shape))
    # Updating word vectors only happens for one verb
    #   So when one verb is done, word vectors should recover
    if p["up_wordvec"]:
        word2vec_bak = np.array(word2vec, copy=True)

    # Get data
    train_loader = DataLoader(
        data_path=p["train_data_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"]
    )
    train, _, validation = train_loader.get_data(
        p["train_part"], 0.0, p["validation_part"],
        sent_num_threshold=p["minimum_sent_num"], 
        frame_threshold=p["minimum_frame"]
    )
    test_loader = DataLoader(
        data_path=p["test_data_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"]
    )
    _, test, _ = test_loader.get_data(
        0.0, p["test_part"], 0.0
    )

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (train data)", 
        "sentence number (test data)", 
        "frame number(test data)",
        'epoch', 'time(m)'
    ]
    scores_overall = np.zeros(len(field_names), dtype=FLOAT)    # Average statistics over all verbs 
    # Class_id to f-score, proportion of occurrence. 
    #   This is very rough because the same class id in different verbs have different meaning
    class_id2scores = {}
    verb_counter = 0
    fh_pr = open(p["prediction_results"], "w")
    verbs = train.keys()
    for verb in verbs:
        verb_counter += 1
        # Recover the word vectors
        if p["up_wordvec"] and verb_counter != 1:
            word2vec = np.array(word2vec_bak, copy=True)
        # Build FNN model for each verb
        fnn = FNN(
            x=train[verb][0], y=train[verb][1], 
            word2vec=word2vec, n_h=p["n_hidden"], 
            n_o=max(train[verb][1]) + 1, weight=p["weight"],
            up_wordvec=p["up_wordvec"]
        )
        if not p["early_stopping"]:
            # Stopping train when zero-one loss is very small on train data
            epoch, time = fnn.train(
                minibatch=p["minibatch"], lr=p["lr"], max_epochs=p["max_epochs"],
                l2_factor=p["l2_factor"], verbose=False
            )
        else:
            # Useing early-stopping
            epoch, time = fnn.early_stopping_train(validation[verb], minibatch=p["minibatch"],
                lr=p["lr"], l2_factor=p["l2_factor"], max_epochs=p["max_epochs"], patience=15,
                improvement_threould=0.995, validation_freq=3, verbose=False
            )

        y_pred = fnn.predict(test[verb][0])
        precision, recall, f_score, class_id2fscore, class_id2prop = standard_score(
            y_true=test[verb][1], y_pred=y_pred
        )
        scores = [
            precision, recall, f_score,
            len(train[verb][1]), 
            len(test[verb][1]), 
            len(set(test[verb][1])),
            epoch, time
        ]
        scores_overall += scores
        print("current verb:%s, scores are:" % verb)
        print(gen_print_info(field_names, scores))
        print("current completeness:%d/%d, average scores over %d verbs are:" %\
                (verb_counter, len(verbs), verb_counter))
        print(gen_print_info(field_names, scores_overall / verb_counter))

        # Compute the sum score of each class
        for class_id in class_id2fscore.keys():
            if class_id not in class_id2scores:
                # f-score, proportion, counter
                class_id2scores[class_id] = [0., 0., 0]
            class_id2scores[class_id][0] += class_id2fscore[class_id]
            class_id2scores[class_id][1] += class_id2prop[class_id]
            class_id2scores[class_id][2] += 1

        # Print prediction results
        sents = indexs2sents(test[verb][0], invocab)
        print("verb: %s\tf-score:%f" % (verb, f_score), file=fh_pr)
        for i in range(0, len(test[verb][1])):
            is_true = True if test[verb][1][i] == y_pred[i] else False
            print("%s\tpredict:%s\ttrue:%s\t%s" %\
                    (is_true, y_pred[i], test[verb][1][i], " ".join(sents[i])), file=fh_pr)

    # File handles
    fhs = [fh_pr, sys.stdout]
    for fh in fhs:
        print(gen_params_info(p), file=fh)
        print("End of training and testing, the average infomation over %d verbs are:" %\
                len(verbs), file=fh)
        print(gen_print_info(field_names, scores_overall / len(verbs)), file=fh)
        print("Detail scores of each class(roughly):", file=fh)
        # Compute the average score of each class
        for class_id in class_id2scores:
            class_id2scores[class_id][0] /= class_id2scores[class_id][2]
            class_id2scores[class_id][1] /= class_id2scores[class_id][2]
            print("class:%d\tf-score:%f\tproportion:%f" %\
                    (class_id, class_id2scores[class_id][0], class_id2scores[class_id][1]), file=fh)
    fh_pr.close()

if __name__ == "__main__":
    run_fnn()

