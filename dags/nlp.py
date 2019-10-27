from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

import urllib
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

# scrape a link and put value in xcom
def scraper(link, **context):
    text =  urllib.request.urlopen("".join(['https://www.sec.gov/Archives/', link])).read().decode()
    context['task_instance'].xcom_push(key="scraped_text", value=text)
    return "pushed text %d" % (len(text))


def preprocess(**context):
    text = context['ti'].xcom_pull(task_ids='scraper', key='scraped_text')
    lines = re.sub("<.*?>", "", text)
    lines = re.sub("&nbsp;", " ", lines)
    lines = re.sub("\s{2,}", "", lines)
    lines = re.sub("^\s+|\s+$", "", lines)
    lines = re.sub("\d+", "", lines)
    lines = re.sub("&*|#*|;*", "", lines)
    lines = lines.split('\n')
    context['task_instance'].xcom_push(key="cleaned_lines", value=lines)
    return "pushed lines %d" % (len(lines))

def sentiment(**context):
    sid = SentimentIntensityAnalyzer()
    lines = context['ti'].xcom_pull(task_ids='preprocess', key='cleaned_lines')
    
    polarity = []
    for line in lines:
        polarity.append(sid.polarity_scores(line))

    context['task_instance'].xcom_push(key='line_polarities', value=polarity)
    return "pushed polarities %d" % (len(polarity))

def polarity_sorter(sort_key, **context):
    polarities = context['ti'].xcom_pull(task_ids='sentiment', key='line_polarities')
    lines = np.array(context['ti'].xcom_pull(task_ids='preprocess', key='cleaned_lines'))
    
    vals = np.array([p[sort_key] for p in polarities])
    sort_idx = np.argsort(vals)[::-1]
    context['ti'].xcom_push(key='top5', value=lines[sort_idx][:5])
    return "Pushed top 5 %s" % (sort_key)

def writer(**context):
    with open('./output.txt', 'w') as f:
        for key in ['pos', 'neg', 'neu']:
            best = context['ti'].xcom_pull(task_ids='_'.join(['sorter', key]), key='top5')
            f.write("Top 5 sentences with %s sentiment\n" % (key))
            f.write(np.array_str(best))
            f.write("\n----------------------------------------------------------------\n\n")
    return "Wrote information to file"

dag = DAG('nlp_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

scraper = PythonOperator(
    task_id="scraper",
    provide_context=True,
    python_callable=scraper,
    dag=dag,
    op_kwargs={'link': 'edgar/data/748015/0001047469-11-000234.txt'}
)

preprocessor = PythonOperator(
        task_id="preprocess",
        provide_context=True,
        python_callable=preprocess,
        dag=dag
)

polarity_analyser = PythonOperator(
        task_id="sentiment",
        provide_context=True,
        python_callable=sentiment,
        dag=dag
)

sorter_neg = PythonOperator(
        task_id="sorter_neg",
        provide_context=True,
        python_callable=polarity_sorter,
        op_kwargs={'sort_key': 'neg'},
        dag=dag
)

sorter_pos = PythonOperator(
        task_id="sorter_pos",
        provide_context=True,
        python_callable=polarity_sorter,
        op_kwargs={'sort_key': 'pos'},
        dag=dag
)

sorter_neu = PythonOperator(
        task_id="sorter_neu",
        provide_context=True,
        python_callable=polarity_sorter,
        op_kwargs={'sort_key': 'neu'},
        dag=dag
)

writer = PythonOperator(
        task_id="writer",
        provide_context=True,
        python_callable=writer,
        dag=dag
)

scraper >> preprocessor >> polarity_analyser >> [sorter_neg, sorter_pos, sorter_neu] >> writer
