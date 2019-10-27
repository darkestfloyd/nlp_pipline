from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import urllib
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import os

from utils import pre_process_document

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime.now(),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 5,
    'retry_delay': timedelta(seconds=5),
}

# scrape a link and put value in xcom
def scraper_fn(link, task_id, **context):
    text =  urllib.request.urlopen("".join(['https://www.sec.gov/Archives/', link])).read().decode()
    context['task_instance'].xcom_push(key="scraped_text", value=text)
    return "pushed text %d" % (len(text))

# preprocess text, code in utils.py file
def preprocess_fn(task_id, **context):
    text = context['ti'].xcom_pull(task_ids='scraper_%d' % task_id, key='scraped_text')
    cleaned_text = pre_process_document(text)
    lines = [l for l in cleaned_text.split('\n')]
    context['task_instance'].xcom_push(key="cleaned_lines", value=lines)
    return "pushed lines %d" % (len(lines))

# use vader to find polarity of each sentence in the document
def sentiment_fn(task_id, **context):
    sid = SentimentIntensityAnalyzer()
    lines = context['ti'].xcom_pull(task_ids='preprocess_%d' % task_id, key='cleaned_lines')
    
    polarity = []
    for line in lines:
        polarity.append(sid.polarity_scores(line))

    context['task_instance'].xcom_push(key='line_polarities', value=polarity)
    return "pushed polarities %d" % (len(polarity))

# sort sentence polarity to make it easy to find most positive and negative sentences
def polarity_sorter_fn(sort_key, task_id, **context):
    polarities = context['ti'].xcom_pull(task_ids='sentiment_%d' % task_id, key='line_polarities')
    lines = np.array(context['ti'].xcom_pull(task_ids='preprocess_%d' % task_id, key='cleaned_lines'))
    
    vals = np.array([p[sort_key] for p in polarities])
    sort_idx = np.argsort(vals)[::-1]
    context['ti'].xcom_push(key='sentiment_scores_order', value=sort_idx)
    context['ti'].xcom_push(key='sentiment_scores', value=vals)
    return "Pushed polarity and order"
    
# write results to output.txt
def writer_fn(outfile, task_id, **context):
    lines = np.array(context['ti'].xcom_pull(task_ids='preprocess_%d' % task_id, key='cleaned_lines'))
    scores = context['ti'].xcom_pull(task_ids='sorter_compound_%d' % task_id, key='sentiment_scores')
    order = context['ti'].xcom_pull(task_ids='sorter_compound_%d' % task_id, key='sentiment_scores_order')

    def _writer(f, i, l, p):
        f.write("Sentence %d" % (i+1))
        f.write('\n')
        f.write(l)
        f.write('\n')
        f.write("Polarity score: %.3f" % p)
        f.write("\n-------------------------------------------------------------------\n\n")


    with open(outfile, 'w') as f:
        f.write("Top 5 sentences with positive sentiment\n")
        for i, (line, polarity) in enumerate(zip(lines[order[:5]], scores[order[:5]])):
            _writer(f, i, line, polarity)
        f.write("\n\nTop 5 sentences with negative sentiment\n")
        for i, (line, polarity) in enumerate(zip(lines[order[-5:][::-1]], scores[order[-5:][::-1]])):
            _writer(f, i, line, polarity)

    return "Wrote information to file %s" % outfile

dag = DAG('nlp_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

input_path = './input.tsv'
input_file = pd.read_csv(input_path, delimiter='|', header=None, 
        names=['cik', 'name', 'filing_type', 'filing_date', 'filing_link', 'filing_details'])

if not os.path.exists('./out_files/'):
    os.makedirs('./out_files/')

def print_init(**context):
    return "init"

starter = PythonOperator(
        task_id="init",
        dag=dag,
        python_callable=print_init,
        provide_context=True
)

ender = PythonOperator(
        task_id="complete",
        dag=dag,
        python_callable=print_init,
        provide_context=True
)

input_file.assign(sentiment_file='')
for idx, row in input_file.iterrows():
    task_id = idx+1
    scraper = PythonOperator(
        task_id="scraper_%d" % task_id,
        provide_context=True,
        python_callable=scraper_fn,
        dag=dag,
        op_kwargs={'link': row['filing_link'], 'task_id': task_id}
    )

    preprocessor = PythonOperator(
            task_id="preprocess_%d" % task_id,
            provide_context=True,
            python_callable=preprocess_fn,
            dag=dag,
            op_kwargs={'task_id':task_id}
    )

    polarity_analyser = PythonOperator(
            task_id="sentiment_%d" % task_id,
            provide_context=True,
            python_callable=sentiment_fn,
            dag=dag,
            op_kwargs={'task_id':task_id}
    )

    sorter_compound = PythonOperator(
            task_id="sorter_compound_%d" % task_id,
            provide_context=True,
            python_callable=polarity_sorter_fn,
            op_kwargs={'sort_key': 'compound', 'task_id': task_id},
            dag=dag,
    )

    outfile_path = './out_files/%s.txt' % ('_'.join([str(row['cik']), row['filing_date']]))

    writer = PythonOperator(
            task_id="writer_%d" % task_id,
            provide_context=True,
            python_callable=writer_fn,
            dag=dag,
            op_kwargs={'outfile': outfile_path, 'task_id': task_id}
    )   

    input_file.loc[idx, 'sentiment_file'] = outfile_path

    starter >> scraper >> preprocessor >> polarity_analyser >> sorter_compound >> writer >> ender

input_file.to_csv('./input_complete.tsv', sep='|')
