#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/29
Brief:  Examples of running baselines
"""

import sys
sys.path.append("../models/lib/")
sys.path.append("../utils/")
sys.path.append("../models/")
from inc import*
from tools import*
from data_loader import DataLoader
from metrics import*
from parameter_setting import*
from baselines import*


def gen_print_info(field_names, values):
    """
    Generate print infomation
    field_names: 1d array-like, each element of field_names is string
    values: 1d array-like. The value of field_names
    return: string
    """
    if len(field_names) != len(values):
        logging.error("The length is not the same.field_names:%s, values:%s"
                      % (str(field_names), str(values)))
        raise Exception
    res = ""
    for i in range(0, len(field_names)):
        res += "%s:%s\t" % (field_names[i], values[i])
    return res


def run_fnn():
    # Parameters
    # p = p_fnn
    p = p_fnn_logic_test
    p["prediction_results"] = "../result/baselines/null"
    # p["minimum_sent_num"] = 100
    # Use MF baseline
    use_mf = False
    rs_set = False

    # Get vocabulary and word vectors
    vocab, invocab, word2vec = get_vocab_and_vectors(
        p["word2vec_path"], norm_only=p["norm_vec"], oov=p["oov"],
        oov_vec_padding=0., dtype=FLOAT, file_format="auto"
    )

    # Get data
    train_loader = DataLoader(
        data_path=p["data_path"], vocab=vocab, oov=p["oov"],
        left_win=p["left_win"], right_win=p["right_win"],
        use_verb=p["use_verb"], lower=p["lower"]
    )
    train, test, validation = train_loader.get_data(
        p["train_part"], p["test_part"], p["validation_part"],
        sent_num_threshold=p["minimum_sent_num"],
        frame_threshold=p["minimum_frame"]
    )

    field_names = [
        'precision', 'recall', 'f-score',
        "sentence number (train data)",
        "sentence number (test data)",
        "frame number(test data)",
        'epoch'
    ]
    # Average statistics over all verbs
    scores_overall = np.zeros(len(field_names), dtype=FLOAT)
    verb_counter = 0
    fh_pr = open(p["prediction_results"], "w")
    verbs = train.keys()
    for verb in verbs:
        verb_counter += 1
        label_y = train[verb][1]
        if use_mf:
            mf = MostFrequent(label_y)
            y_pred = mf.select(len(test[verb][0]))
        else:
            rs = RandomSelector(label_y)
            y_pred = rs.select_array(len(test[verb][0]), rs_set)

        precision, recall, f_score, _, _ = standard_score(
            y_true=test[verb][1], y_pred=y_pred
        )
        epoch = 0
        scores = [
            precision, recall, f_score,
            len(train[verb][1]),
            len(test[verb][1]),
            len(set(test[verb][1])),
            epoch
        ]
        scores_overall += scores
        print("current verb:%s, scores are:" % verb)
        print(gen_print_info(field_names, scores))
        print("current completeness:%d/%d, average scores over %d verbs are:"
              % (verb_counter, len(verbs), verb_counter))
        print(gen_print_info(field_names, scores_overall / verb_counter))

        # Print prediction results
        sents = indexs2sents(test[verb][0], invocab)
        print("verb: %s\tf-score:%f" % (verb, f_score), file=fh_pr)
        for i in range(0, len(test[verb][1])):
            is_true = True if test[verb][1][i] == y_pred[i] else False
            print("%s\tpredict:%s\ttrue:%s\t%s"
                  % (is_true, y_pred[i], test[verb][1][i],
                     " ".join(sents[i])), file=fh_pr)

    # File handles
    fhs = [fh_pr, sys.stdout]
    for fh in fhs:
        print(gen_params_info(p), file=fh)
        print("End of training and testing, the average "
              "infomation over %d verbs are:" % len(verbs), file=fh)
        print(gen_print_info(field_names, scores_overall / len(verbs)),
              file=fh)
    fh_pr.close()

if __name__ == "__main__":
    run_fnn()
