import json
import re
#from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize
from num2word import num2word
def my_replace(match):
    match = match.group()
    try:
        return ' '+num2word(round(float(match)))+' '
    except Exception:
        return ' '

sentences = open('sentences.txt','a')
with open('../News_articles/cc_news_en.jsonlist') as infile:
    for line in infile:
        news = json.loads(line)
        if news['text']!=None:
            for sent in sent_tokenize(re.sub(r'http\S+|[^\x00-\x7f]','', news['text'])):
                s=re.sub(r'[0-9]+[\.[0-9]+$]*', my_replace, " ".join(re.split('\W+|_', sent)).lower())
                sentences.write(s+'\n')
        elif news['description']!=None:
            for sent in sent_tokenize(re.sub(r'http\S+|[^\x00-\x7f]','', news['description'])):
                s=re.sub(r'[0-9]+[\.[0-9]+$]*', my_replace, " ".join(re.split('\W+|_', sent)).lower())
                sentences.write(s+'\n')
sentences.close()
