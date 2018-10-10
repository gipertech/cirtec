# -*- coding: utf-8 -*-

def get_average_len_words(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)
