# -*- coding: utf-8 -*-

""""""


from gensim.models.phrases import Phrases


def get_average_len_words(sentence):
    words = sentence.split()
    if len(words) == 0:
        return 0
    return sum([len(word) for word in words]) / len(words)


def get_n_gram(sentences, n, delimiters=(b"@", b"#", b"$", b"%"), min_count=20, threshold=10):
    if n < 2:
        raise ValueError(" n < 2 ")
    if len(delimiters) < n - 1:
        raise ValueError(" len(delimiters) < n-1 ")
    grams = []
    for ind in range(n - 1):
        gram = Phrases(sentences, min_count=min_count, delimiter=delimiters[ind], threshold=threshold)
        grams.append(gram)
        if ind != n - 2:
            sentences = gram[sentences]
    return grams


def drop_all_delimiters(text, delimiters):
    for delimiter in delimiters:
        text = text.replace(delimiter, " ")
    return text


def get_gram_vocab(grams, n, delimiters=("@", "#", "$", "%")):
    if n < 2:
        raise ValueError(" n < 2 ")
    if len(delimiters) < n - 1:
        raise ValueError(" len(delimiters) < n-1 ")
    delimiters = delimiters[:n - 1]
    sorted_gram_vocab = sorted([(drop_all_delimiters(value, delimiters), count) for (value, count) in
                                [(value.decode('utf8'), count) for value, count in dict(grams[n - 2].vocab).items()]
                                if sum([dlm in value for dlm in delimiters]) == n - 1], key=lambda kv: kv[1],
                               reverse=True)

    return sorted_gram_vocab
