#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016/03/23
Brief:  Convert pdev data and SemEval 2015 data to the required format
"""

import os
import html
import nltk
from tools import*
import logging
logging.basicConfig(
    level=logging.DEBUG, 
    format=" [%(levelname)s]%(filename)s:%(lineno)s[function:%(funcName)s] %(message)s"
)

def convert_semeval_without_extraction(detail=True):
    """
    Convert semeval data without extraction the arugments of target verb
    """

    # Parameters
    p = {
        "data_sets": ["../cpa_data/Microcheck/", "../cpa_data/testdata/Microcheck"], 
        "output_dir": "../data/semeval_mic_train_and_test_no_extraction"
    }
    if detail:
        print_params(p)
    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    for data_set_path in p["data_sets"]:
        files = os.listdir("%s/input/" % data_set_path)
        for file_name in files:
            if detail:
                print("to process %s" % file_name)
            verb_name, ext = os.path.splitext(file_name)
            file_path = "%s/input/%s" % (data_set_path, file_name)
            # Read cluster
            file_cluster_path = "%s/task2/%s.clust" % (data_set_path, verb_name)
            fh = open(file_cluster_path, "r")
            verb_id2cluster_id = {}
            for line in fh:
                line = line.strip()
                if line == "":
                    continue
                verb_id, cluster_id = line.split("\t")
                verb_id2cluster_id[verb_id] = cluster_id
            fh.close()

            out_file_path = "%s/%s" % (p["output_dir"], verb_name)
            fh_out = open(out_file_path, "w")
            fh = open(file_path, "r", encoding = "ISO-8859-1")
            sent = ""
            cluster_id = -1
            for line in fh:
                line = line.strip()
                if line == "":
                    fh_out.write("%s\t%s\n" % (cluster_id, remove_punctuations(sent)))
                    sent = ""
                    continue
                tokens = line.split("\t")
                if len(tokens) == 3 and tokens[2] == "v":
                    sent += "\t%s\t" % tokens[1]
                    cluster_id = verb_id2cluster_id[tokens[0]]
                else:
                    sent += tokens[1] + " "
            fh.close()
            fh_out.close()

def convert_semeval_with_extraction(detail=True):
    """
    Convert semeval data with extraction the arugments of target verb
    """

    # Parameters
    p = {
        "data_sets": ["../cpa_data/Microcheck/", "../cpa_data/testdata/Microcheck/"], 
        "output_dir": "../data/semeval_mic_train_and_test_with_extraction", 
        "normalized_units": False, 
        "relations": ["subj", "obj", "iobj", "advprep", "acomp", "scomp"], 
        "padding": "_"  # Used for normalized_units
    }
    if detail:
        print_params(p)
    os.system("rm -rf %s" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    for data_set_path in p["data_sets"]:
        files = os.listdir("%s/task1/" % data_set_path)
        for file_name in files:
            if detail:
                print("to process %s" % file_name)
            verb_name, ext = os.path.splitext(file_name)
            file_path = "%s/task1/%s" % (data_set_path, file_name)
            # Read cluster
            file_cluster_path = "%s/task2/%s.clust" % (data_set_path, verb_name)
            fh = open(file_cluster_path, "r")
            verb_id2cluster_id = {}
            for line in fh:
                line = line.strip()
                if line == "":
                    continue
                verb_id, cluster_id = line.split("\t")
                verb_id2cluster_id[verb_id] = cluster_id
            fh.close()

            out_file_path = "%s/%s" % (p["output_dir"], verb_name)
            fh_out = open(out_file_path, "w")
            fh = open(file_path, "r", encoding = "ISO-8859-1")
            if not p["normalized_units"]:
                sent = ""
                cluster_id = -1
                for line in fh:
                    line = line.strip()
                    if line == "":
                        fh_out.write("%s\t%s\n" % (cluster_id, sent))
                        sent = ""
                        continue
                    tokens = line.split("\t")
                    if len(tokens) == 4 and tokens[2] == "v":
                        sent += "\t%s\t" % tokens[1]
                        cluster_id = verb_id2cluster_id[tokens[0]]
                    else:
                        sent += tokens[1] + " "
            else:
                syn2word = {}
                for line in fh:
                    line = line.strip()
                    if line == "":
                        fh_out.write("%s\t" % cluster_id)
                        for i in range(0, len(p["relations"])):
                            relation = p["relations"][i]
                            # Insert verb and cluster id in the middle
                            if i == int(len(p["relations"]) / 2):
                                fh_out.write("\t%s\t" % verb)
                            if relation not in syn2word:
                                fh_out.write(" %s" % p["padding"])
                                continue
                            fh_out.write(" %s" % syn2word[relation])
                        fh_out.write("\n")
                        continue
                    tokens = line.split("\t")
                    if len(tokens) == 4:
                        if tokens[2] == "v":
                            verb = tokens[1]
                            cluster_id = verb_id2cluster_id[tokens[0]]
                        else:
                            syn2word[tokens[2]] = tokens[1]
            
            fh.close()
            fh_out.close()

def convert_pdev(detail=True):
    """
    Convert pdev data
    """
    p = {
        "pdev_dir": "../split_pdev/train_pdev/", 
        "output_dir": "../data/split_pdev/train"
    }
    os.system("rm %s -rf" % p["output_dir"])
    os.system("mkdir -p %s" % p["output_dir"])
    files = os.listdir(p["pdev_dir"])
    for file_name in files:
        file_path = "%s/%s" % (p["pdev_dir"], file_name)
        fh_in = open(file_path, "r")
        out_file = "%s/%s" % (p["output_dir"], file_name)
        fh_out = open(out_file, "w")
        for line in fh_in:
            line = line.strip()
            if line == "":
                continue
            line = html.unescape(line).replace("<p>", " . ").replace("</p>", " . ")
            items = line.split("\t")
            if len(items) != 4:
                continue
            frame = items[2].strip()
            # Processing frame
            frame = frame.split(".")[0]
            try:
                frame = int(frame)
            except:
                logging.warn("Skip frame: %s" % frame) 
                continue
            verb = items[1].strip()
            left_sent = remove_punctuations(nltk.sent_tokenize(items[0])[-1])
            right_sent = remove_punctuations(nltk.sent_tokenize(items[3])[0])
            # Remove corpus title infomation
            right_sent = right_sent.replace("British National Corpus", "")
            fh_out.write("%d\t%s\t%s\t%s\n" % (frame, left_sent, verb, right_sent))
        fh_in.close()
        fh_out.close()

if __name__ == "__main__":
    # convert_semeval_without_extraction()
    # convert_semeval_with_extraction()
    convert_pdev()

