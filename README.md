# cirtec
CIRTEC project, user `tuzovsky`

# clustering_analysis.py
Создаёт файл json формата содержащий аналитику компакности кластеров

Для получения справки по использованию скрипта используется команда: `python clustering_analysis.py -h`
## Usage
```
clustering_analysis.py [-h] --in-filename IN_FILENAME --out-filename OUT_FILENAME --model MODEL

optional arguments:
  -h, --help            show this help message and exit
  --in-filename IN_FILENAME
                        Path to the input file
  --out-filename OUT_FILENAME
                        Path to the output file
  --model MODEL         Path to the w2v model
```
## Example
```
python clustering_analysis.py --model ./models/fixes.stem.cbow.bin --in-filename ./initial_data/Word2Vec__citcon4bundles.txt --out-filename clustering.json
```


# create_json_grams.py
Создаёт файл json формата содержащий аналитику по N-граммам

Для получения справки по использованию скрипта используется команда: `python clustering_analysis.py -h`

# Usage
```
create_json_grams.py [-h] --in-filename IN_FILENAME --out-filename OUT_FILENAME

optional arguments:
  -h, --help            show this help message and exit
  --in-filename IN_FILENAME
                        Path to the input file
  --out-filename OUT_FILENAME
                        Path to the output file
```
## Example
```
python create_json_grams.py --in-filename ./initial_data/Word2Vec__citcon4bundles.txt --out-filename grams.json
```
