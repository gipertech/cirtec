# -*- coding: utf-8 -*-

""""""

# Author: Bedarev Nickolay
# Mail: n.bedarev@lcgroup.su
# Limeteam slack: nikkollaii
# Date: 07.10.2018


def get_average_len_words(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)
