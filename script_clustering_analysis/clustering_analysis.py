# -*- coding: utf-8 -*-

""""""

# Author: Bedarev Nickolay
# Mail: n.bedarev@lcgroup.su
# Limeteam slack: nikkollaii
# Date: 27.11.2018

import re
import json
import gensim
import argparse
import pymorphy2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances

morph = pymorphy2.MorphAnalyzer()
regex = re.compile('[^а-яА-Я]')


def get_average_len_words(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)


def get_word_with_correct_tag(word):
    pars_res = morph.parse(word)[0]
    return pars_res.normal_form


def load_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    text = data.decode('utf-8')  # a `str`; this step can't be used if data is binary
    del data
    sentences = [(sentence[0], sentence[1], sentence[2], sentence[3], sentence[4].replace("\xad ", "").lower())
                 for sentence in (sentence.split(" ", 4) for sentence in text.split("\n") if
                                  sentence and get_average_len_words(sentence) > 5)]
    del text
    print("len_sentences: {}".format(len(sentences)))
    list_sentences = [{
        "name": sentence[0],
        "id": " ".join([sentence[0], sentence[1], sentence[2], sentence[3]]),
        "sentence": sentence[4],
        "words_normal_form": [get_word_with_correct_tag(word) for word in regex.sub(' ', sentence[4]).split() if
                              word not in ["й", "ч", "р"]],
    } for sentence in sentences]
    del sentences
    return [line for line in list_sentences if len(line["words_normal_form"]) > 7]


def load_model(model_path, binary=True, unicode_errors='ignore', init_sims=True):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary, unicode_errors=unicode_errors)
    if init_sims:
        w2v_model.init_sims(replace=True)
    return w2v_model


def get_word_averaging(words, w2v_model):
    mean = []

    for word in words:
        if word in w2v_model.vocab:
            mean.append(w2v_model.vectors_norm[w2v_model.vocab[word].index])

    if not mean:
        return np.zeros(w2v_model.vector_size)

    mean = gensim.matutils.unitvec(np.average(mean, axis=0)).astype(np.float32)
    return mean


def get_average_value(lines):
    mat = pairwise_distances([line["words_averaging"] for line in lines], metric="cosine")
    return np.average(mat[np.argmin([sum(l) for l in mat])])


def get_average_value_in_clusters(lines):
    res = dict()

    for ind in set([line["cluster_id"] for line in lines]):
        mat = pairwise_distances([line["words_averaging"] for line in lines if line["cluster_id"] == ind],
                                 metric="cosine")
        res[ind] = np.average(mat[np.argmin([sum(l) for l in mat])])

    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""""")

    parser.add_argument('--in-filename', type=str, required=True,
                        help="Path to the input file")
    parser.add_argument('--out-filename', type=str, required=True,
                        help="Path to the output file")

    parser.add_argument('--model', type=str, required=True,
                        help="Path to the w2v model")

    start_args = parser.parse_args()

    IN_FILENAME = start_args.in_filename
    MODEL = start_args.model
    OUT_FILENAME = start_args.out_filename

    model = load_model(MODEL)
    list_sentences = load_file(IN_FILENAME)

    for line in list_sentences:
        line["words_averaging"] = get_word_averaging(line["words_normal_form"], model)

    all_names = set([line["name"] for line in list_sentences])
    print(f"all_names: {len(all_names)}")

    for name in all_names:
        lines = [line for line in list_sentences if line["name"] == name]
        for pair in zip(lines, DBSCAN(eps=0.045, min_samples=2, metric="cosine").fit_predict(
                [line["words_averaging"] for line in lines])):
            pair[0]["cluster_id"] = pair[1]

    res_dict = dict()

    for name in all_names:
        lines = [line for line in list_sentences if line["name"] == name]
        average_value_in_clusters = get_average_value_in_clusters(lines)

        res_dict[name] = dict()
        res_dict[name]["average_to_clustroid"] = float(get_average_value(lines))
        res_dict[name]["internal_clusters"] = dict()
        for cluster_id in set([line["cluster_id"] for line in lines]):
            cluster = [line["id"] for line in lines if line["cluster_id"] == cluster_id]
            res_dict[name]["internal_clusters"][str(cluster_id)] = {
                "lines": cluster,
                "size": len(cluster),
                "average_to_clustroid": float(average_value_in_clusters[cluster_id]),
            }

    with open(OUT_FILENAME, "w") as f:
        print(f"OUT: {OUT_FILENAME}")
        json.dump(res_dict, f)
