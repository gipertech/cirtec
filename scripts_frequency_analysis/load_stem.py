import urllib.request
url = 'http://cirtec.ranepa.ru/Word2Vec/fixes.stem.txt'
response = urllib.request.urlopen(url)
data = response.read()      # a `bytes` object

with open('../initial_data/Word2Vec__fixes.stem.txt', 'wb') as f:
    f.write(data)
