# https://github.com/dipanjanS/data_science_for_all/blob/master/tds_deep_transfer_learning_nlp_classification/Deep%20Transfer%20Learning%20for%20NLP%20-%20Text%20Classification%20with%20Universal%20Embeddings.ipynb

import numpy as np
import contractions
from bs4 import BeautifulSoup
import unicodedata
import re


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text):
    return contractions.fix(text)

# def remove_special_characters(text, remove_digits=False):
#     pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
#     text = re.sub(pattern, '\s', text)
#     return text


def pre_process_document(document):
    
    # strip HTML
    document = strip_html_tags(document)
    
    # lower case
    document = document.lower()
    
    # remove extra newlines (often might be present in really noisy text)
    # document = document.translate(document.maketrans("\n\t\r", "   "))
    
    # remove accented characters
    document = remove_accented_chars(document)
    
    # expand contractions    
    document = expand_contractions(document)
               
    # remove special characters and\or digits    
    # insert spaces between special characters to isolate them    
    # special_char_pattern = re.compile(r'([{.(-)!}])')
    # document = special_char_pattern.sub(" \\1 ", document)
    # document = remove_special_characters(document, remove_digits=True)  
    document = re.sub('[^A-Za-z\n]+', ' ', document)

    # remove extra whitespace
    #document = re.sub(' +', ' ', document)
    #document = document.strip()
    
    return document


pre_process_corpus = np.vectorize(pre_process_document)
