# coding=utf-8
# python lda.py KankerNL_threads

# The input directory is expected to contain a list of XML files in the unified forum XML format, as created by json2xml_iknl.py (see dtd below)
# The LDA model is printed to stdout

"""
<!ELEMENT thread (threadid,title,post+,category*,type*,nrofviews?)>
<!ELEMENT post (postid,author,timestamp,parent*,upvotes?,downvotes?,body)>
<!ELEMENT author (#PCDATA)>
<!ELEMENT timestamp (#PCDATA)>
<!ELEMENT parent (#PCDATA)>
<!ELEMENT upvotes (#PCDATA)>
<!ELEMENT downvotes (#PCDATA)>
<!ELEMENT body (content,url*)>
<!ELEMENT content (#PCDATA)>
<!ELEMENT url (#PCDATA)>

"""

from gensim import corpora
from gensim import models
#import logging
import re
import sys
import xml.etree.ElementTree as ET
import os
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(message)s', level=logging.INFO)


inputdir = sys.argv[1]


documents_per_category = dict()
documents = []

def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds


for inputfile in os.listdir(inputdir):
    if inputfile.endswith("xml"):
        with open (inputdir+"/"+inputfile,'r') as xml_file:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for thread in root:
                threadid = thread.get('id')
                #print (threadid)
                title = thread.find('title').text
                for posts in thread.findall('posts'):
                    for post in posts.findall('post'):
                        postid = post.get('id')
                        bodyofpost = post.find('body').text
                        if bodyofpost is None:
                            bodyofpost = ""
                        if re.match(".*http://[^ ]+\n[^ ]+.*",bodyofpost):
                            bodyofpost = re.sub("(http://[^ ]+)\n([^ ]+)",r"\1\2",bodyofpost)
                            bodyofpost = re.sub("[^ ]*http://[^ ]+","",bodyofpost)
                        documents.append(bodyofpost)


# remove common words and tokenize
stoplist = set()
with open(r'stoplist_dutch.txt') as stoplist_file:
    for line in stoplist_file:
        stopword = line.rstrip()
        stoplist.add(stopword)



def prepare_corpus(documents):
    print ("Prepare corpus of",len(documents),"documents")
    # a word should be at least three letters and not a stopword
    texts = [[word for word in tokenize(document) if word not in stoplist and re.match(".*[a-z][a-z][a-z].*",word)]
             for document in documents]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    #print(texts)

    dictionary = corpora.Dictionary(texts)
    #print(dictionary)
    print ("Dictionary size:",len(dictionary))
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print(corpus)
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return corpus_tfidf,dictionary



def create_topic_model(documents,num_topics,num_words):
    corpus,dictionary = prepare_corpus(documents)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False, num_words=num_words)

    for topic in topics:
        print ("++",topic[0],":",topic[1])

'''
MAIN
'''

num_topics = 50
num_words = 10
create_topic_model(documents,num_topics,num_words)


