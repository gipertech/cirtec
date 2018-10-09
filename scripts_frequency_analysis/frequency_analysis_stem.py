import pandas as pd

from base_defs import get_average_len_words, get_n_gram, drop_all_delimiters, get_gram_vocab

with open('../initial_data/Word2Vec__fixes.stem.txt', 'rb') as f:
    data = f.read()
text = data.decode('utf-8') # a `str`; this step can't be used if data is binary
del data
print("len_text: {}".format(len(text)))
sentences = [sentence.lower() for sentence in text.split("\n") if len(sentence) > 60]
del text
print("len_sentences: {}".format(len(sentences)))
print(sentences[0])

min_average_len_words = 9
sentences = [[word for word in sentence.split()] for sentence in sentences
             if get_average_len_words(sentence) >= min_average_len_words]
print(len(sentences))
print(sentences[0][:5])

grams = get_n_gram(sentences=sentences, n=4, delimiters=(b"@", b"#", b"$", b"%"))

del sentences

grams_vocab = []
for ind in range(2, 5):
     grams_vocab.extend(get_gram_vocab(grams=grams, n=ind, delimiters=("@", "#", "$", "%")))
res = pd.DataFrame(grams_vocab, columns=["text", "freq"]).drop_duplicates(["text"]).sort_values(["freq"], ascending=False).reset_index(drop=True)
res["len_text"] = res.text.apply(lambda x: len(x.split()))
res['text_without_tags'] = res['text'].apply(lambda x: " ".join([word.split("_")[0] for word in x.split()]))

def save_csv(res_df, path, len_text):
    filename = "{}/frequency_analysis_stem_{}.csv".format(path, len_text)
    print(filename)
    res_df[res_df.len_text == len_text][["text_without_tags", "freq"]].to_csv(filename, index=False)

save_csv(res, path="../resulting_data", len_text=2)
save_csv(res, path="../resulting_data", len_text=3)
save_csv(res, path="../resulting_data", len_text=4)
