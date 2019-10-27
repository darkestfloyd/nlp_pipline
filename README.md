# Reproducible NLP pipeline using Airflow
Demonstration of reproducible sentiment analysis pipeline using Airflow

### Set up airflow and code
To run, make sure airflow is installed. You will also need the NLTK package with vader_lexicon.

To download vader_lexicon, run:

`python -c "import nltk; nltk.download('vader_lexicon')"`

```
# in a terminal
git clone git@github.com:nischalchand/nlp_pipline.git
mv nlp_pipeline ~/airflow

# init airflow database
airflow initdb

# in new terminal
airflow webserver -p 8080

# in another new terminal
airflow scheduler
```

### Set up input
An `input.tsv` file is included in the repo, you can use [pyhton-edgar](https://pypi.org/project/python-edgar/) to get more data.

To filter out only 10k filings from edgar download files, in terminal 

```
cd <path to downloaded files by python-edgar>
grep -h 10-K * > 10k.tsv
cp 10k.tsv ~/airflow/input.tsv
```

Go to `localhost:8080` in web browser, turn-on and trigger "nlp_pipeline". The number of branches in the DAG will be dependent on the number of rows in `input.tsv`.

### Output information
An `input_complete.tsv` file is created in the same directory, which is a copy of `input.tsv` with an additional 
column `sentiment_file` for the path of the sentiment file. Each sentiment file is stored in `out_files`, with the file format 
`<cik>_<date>.txt`. 
