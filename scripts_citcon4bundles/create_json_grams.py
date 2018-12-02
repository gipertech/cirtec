# -*- coding: utf-8 -*-

""""""

# Author: Bedarev Nickolay
# Mail: n.bedarev@lcgroup.su
# Limeteam slack: nikkollaii
# Date: 30.11.2018

import re
import json
import argparse
import pymorphy2

from collections import Counter
from gensim.models.phrases import Phrases

morph = pymorphy2.MorphAnalyzer()
regex = re.compile('[^а-яА-Я]')

DELIMITER = "_"
DELIMITER_B = b"_"


def get_average_len_words(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)


def load_data(filename: str):
    with open(filename, 'rb') as f:
        data = f.read()
    text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
    del data
    sentences = [(sentence[0], sentence[1], sentence[2], sentence[3], sentence[4].replace("\xad ", "").lower())
                 for sentence in (sentence.split(" ", 4) for sentence in text.split("\n") if sentence and get_average_len_words(sentence) > 5)]
    del text

    words = [(sentence[0], [word for word in regex.sub(' ', sentence[4]).split()]) for sentence in sentences]
    words_normal_form = [(sentence[0], [morph.parse(word)[0].normal_form for word in regex.sub(' ', sentence[4]).split()]) for sentence in sentences]

    del sentences

    words = [(pair[0], [word for word in pair[1] if word not in ["й", "ч", "р", "ил"]]) for pair in words]
    words_normal_form = [(pair[0], [word for word in pair[1] if word not in ["й", "ч", "р", "ил"]]) for pair in words_normal_form]
    return words, words_normal_form


def get_n_gram(_words, n, min_count=30, threshold=50):
    if n < 2:
        raise ValueError(" n < 2 ")
    grams = []
    for ind in range(n - 1):
        gram = Phrases(_words, min_count=min_count, delimiter=DELIMITER_B, threshold=threshold)
        grams.append(gram)
        if ind != n - 2:
            _words = gram[_words]
    return grams


def drop_all_delimiters(text):
    text = text.replace(DELIMITER, " ")
    return text


def get_words_with_phrases(pairs, _list_grams):
    _list_words = [pair[1] for pair in pairs]
    for gram in _list_grams:
        _list_words = gram[_list_words]
    return list(zip([pair[0] for pair in pairs], _list_words))


def get_phrases(pairs, n):
    _phrases = [pair[1] for pair in pairs]
    _phrases = [[word for word in sent if len(drop_all_delimiters(word).split()) == n] for sent in _phrases]
    return list(zip([pair[0] for pair in pairs], _phrases))


def get_phrases_more(pairs, n):
    _phrases = [pair[1] for pair in pairs]
    _phrases = [[word for word in sent if len(drop_all_delimiters(word).split()) > n] for sent in _phrases]
    return list(zip([pair[0] for pair in pairs], _phrases))


def get_dict_counters_by_bunshid(pairs):
    _res = dict()
    for name_pair in all_pairs_names:
        all_pair = []
        for _words in (pair[1] for pair in pairs if pair[0] == name_pair):
            all_pair.extend(_words)
        _res[name_pair] = Counter(all_pair)
    return _res


def get_res_dict(__dict_counters, __pair_name, __n, __grams_counter):
    return [
        {
            "w": drop_all_delimiters(counter_words[0]),
            "n": counter_words[1],
            "t": __grams_counter[counter_words[0]]
        } for counter_words in __dict_counters[__n][__pair_name].most_common()
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument('--in-filename', type=str, required=True,
                        help="Path to the input file")
    parser.add_argument('--out-filename', type=str, required=True,
                        help="Path to the output file")

    start_args = parser.parse_args()

    IN_FILENAME = start_args.in_filename
    OUT_FILENAME = start_args.out_filename

    words, words_normal_form = load_data(IN_FILENAME)
    N = 6
    grams = get_n_gram(_words=[word[1] for word in words], n=N)
    grams_normal_form = get_n_gram(_words=[word[1] for word in words_normal_form], n=N)

    words_with_phrases = get_words_with_phrases(words, grams)
    words_normal_form_with_phrases = get_words_with_phrases(words_normal_form, grams_normal_form)

    phrases_words = {i: get_phrases(words_with_phrases, i) for i in range(2, N+1)}
    phrases_words["more"] = get_phrases_more(words_with_phrases, N)

    phrases_words_normal_form = {i: get_phrases(words_normal_form_with_phrases, i) for i in range(2, N+1)}
    phrases_words_normal_form["more"] = get_phrases_more(words_normal_form_with_phrases, N)

    all_pairs_names = set([word[0] for word in words])

    dict_counters_words = {key: get_dict_counters_by_bunshid(pairs) for key, pairs in phrases_words.items()}
    dict_counters_words_normal_form = {key: get_dict_counters_by_bunshid(pairs) for key, pairs in phrases_words_normal_form.items()}

    grams_counter = Counter()
    for gram in [grams[-1]]:
        t = {k.decode("utf-8"): v for k, v in gram.vocab.items()}
        grams_counter.update(Counter({key: value for key, value in t.items() if "_" in key}))

    grams_counter_normal_form = Counter()
    for gram in [grams_normal_form[-1]]:
        t = {k.decode("utf-8"): v for k, v in gram.vocab.items()}
        grams_counter_normal_form.update(Counter({key: value for key, value in t.items() if "_" in key}))

    result_dict = {
        pair_name: {
            "2orig": get_res_dict(dict_counters_words, pair_name, 2, grams_counter),
            "2norm": get_res_dict(dict_counters_words_normal_form, pair_name, 2, grams_counter_normal_form),

            "3orig": get_res_dict(dict_counters_words, pair_name, 3, grams_counter),
            "3norm": get_res_dict(dict_counters_words_normal_form, pair_name, 3, grams_counter_normal_form),

            "4orig": get_res_dict(dict_counters_words, pair_name, 4, grams_counter),
            "4norm": get_res_dict(dict_counters_words_normal_form, pair_name, 4, grams_counter_normal_form),

            "5orig": get_res_dict(dict_counters_words, pair_name, 5, grams_counter),
            "5norm": get_res_dict(dict_counters_words_normal_form, pair_name, 5, grams_counter_normal_form),

            "6orig": get_res_dict(dict_counters_words, pair_name, 6, grams_counter),
            "6norm": get_res_dict(dict_counters_words_normal_form, pair_name, 6, grams_counter_normal_form),

            "more_orig": get_res_dict(dict_counters_words, pair_name, "more", grams_counter),
            "more_norm": get_res_dict(dict_counters_words_normal_form, pair_name, "more", grams_counter_normal_form),
        } for pair_name in all_pairs_names
    }

    with open(OUT_FILENAME, "w") as f:
        json.dump(result_dict, f)
