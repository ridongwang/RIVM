# python lda.py AYA_messages.nonewlines.csv
# python lda.py /Users/suzanverberne/PycharmProjects/DISCOSUMO/dataconversion/Viva_forum/samples/kankerthreads_all.xml lda.viva.kanker.out
# python lda.py /Users/suzanverberne/PycharmProjects/RIVM/forum_threads/all_KankerNL_threads.xml lda.kankerNL.out

from gensim import corpora
from gensim import models
import csv
#import logging
import operator
import re
import sys
import xml.etree.ElementTree as ET
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(message)s', level=logging.INFO)


inputfile = sys.argv[1]
outputfile = sys.argv[2]
out = open(outputfile,'w')

documents_per_category = dict()
documents = []

def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds

if re.match(".*\.csv$",inputfile):
    with open(inputfile,'r') as csv_file:
        data = csv.DictReader(csv_file, delimiter=',', quotechar='"')
        for row in data:
            author = row["PosterID"]
            content = row["MessageText"]
            documents_for_category = []
            if author in documents_per_category:
                # author as category
                documents_for_category = documents_per_category[author]
            documents_for_category.append(content)
            documents_per_category[author] = documents_for_category
            documents.append(content)


elif re.match(".*\.xml$",inputfile):
    with open (inputfile,'r') as xml_file:
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
                    if "smileys" in bodyofpost:
                        bodyofpost = re.sub(r'\((http://forum.viva.nl/global/(www/)?smileys/.*.gif)\)','',bodyofpost)
                    if "kanker" in bodyofpost:
                        #print (bodyofpost)
                        words = tokenize(bodyofpost)
                        for word in words:
                            word = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", word)
                            if re.match(".*kanker$",word):
                                #print ("Kankertype:",word)
                                if 9 <= len(word) <= 25:
                                    # don't include 'kanker, or long URLs ending in kanker
                                    kankertype = word
                                    documents_for_category = []
                                    if kankertype in documents_per_category:
                                        # kankertype as category
                                        documents_for_category = documents_per_category[kankertype]
                                    documents_for_category.append(bodyofpost)
                                    documents_per_category[kankertype] = documents_for_category
                    documents.append(bodyofpost)


remove_categories = dict()
# remove categories with too few documents or too few words in total
for category in documents_per_category:
    #print (category)
    documents_for_category = documents_per_category[category]
    allcontent = ""
    for doc in documents_for_category:
        allcontent += " "+doc
    wordcount_for_category = len(tokenize(allcontent))
    if wordcount_for_category < 50000 or len(documents_for_category) < 100:
        remove_categories[category] = 1
        print ("remove",category,"because of small corpus size")

for cat in remove_categories:
    del(documents_per_category[cat])


# remove common words and tokenize
#stoplist = set('for a of the and to in'.split())
stoplist = set()
with open(r'stoplist_dutch.txt') as stoplist_file:
    for line in stoplist_file:
        stopword = line.rstrip()
        stoplist.add(stopword)


#texts = [[word for word in document.lower().split() if word not in stoplist]
#         for document in documents]

def prepare_corpus(documents):
    print ("Prepare corpus (",len(documents)," documents)")
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

def create_topic_model(corpus_tfidf,dictionary,category_name,numtopics):
    #lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
    print("Create topic model for category",category_name)
    lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=numtopics)
    #lsi.print_topics(10)
    #lda.print_topics(50)
    return lda


def print_topic_model (category_name,numtopics,model):
    out.write("\n\n>>>LDA-model voor "+category_name+"<<<\n")
    #lda.print_topics(num_topics=50, num_words=10)
    topics= model.show_topics(num_topics=numtopics, formatted=False, num_words=5)
    i=0
    for topic in topics:
        i += 1
        print (topic)
        #topicid = topic[0]
        wordswithprobs = topic[1]
        out.write("Topic #"+str(i)+":")
        for (word,prob) in wordswithprobs:
            out.write(" "+word+ "("+str(prob)+"),")
        out.write("\n")



def get_t2_with_maximum_overlap_to_t1 (topic1,t1,topics2,overlapping_wordswithprobs_per_t1_t2):
    wordswithprobs1 = topic1[1]
    t2 = 0
    maximum_overlap_size = 0
    wordswithprobs_with_maximum_overlap = list()
    id_of_topic_with_maximum_overlap = ""

    # first get the topic from run2 with the maximum overlap to this topic from run1
    for topic2 in topics2:
        t2 += 1

        #print ("find overlap between topic",t1,"from run1 and topic",t2,"from run2")
        overlapping_wordswithprobs = list()
        wordswithprobs2 = topic2[1]
        for (word1,prob1) in wordswithprobs1:
            #print (word)
            for (word2,prob2) in wordswithprobs2:
                if word1 == word2:
                    overlapping_wordswithprobs.append((word1,prob1))
                    #print ("overlapping word in topic",i1,"from run1 and topic",i2,"from run2:", word1)

        if len(overlapping_wordswithprobs) > maximum_overlap_size:
            wordswithprobs_with_maximum_overlap = overlapping_wordswithprobs
            id_of_topic_with_maximum_overlap = t2
            maximum_overlap_size = len(overlapping_wordswithprobs)
            overlapping_wordswithprobs_per_t1_t2[(t1,t2)] = overlapping_wordswithprobs

    return wordswithprobs_with_maximum_overlap, id_of_topic_with_maximum_overlap, maximum_overlap_size, overlapping_wordswithprobs_per_t1_t2


def find_stable_topics(topics1,topics2):

    t1 = 0
    overlapping_topics_per_t2 = dict()
    # key is t2, value is list of t1 topics that have overlap with t2
    overlapping_wordswithprobs_per_t1_t2 = dict()
    for topic1 in topics1:
        t1 += 1

        # for each t1, find the topic t2 with the maximum overlap:
        wordswithprobs_with_maximum_overlap,id_of_topic_with_maximum_overlap, maximum_overlap_size, overlapping_wordswithprobs_per_t1_t2 = get_t2_with_maximum_overlap_to_t1(topic1,t1,topics2,overlapping_wordswithprobs_per_t1_t2)

        if len(wordswithprobs_with_maximum_overlap) > 1:
            # only keep overlap of at least 2 words (single-word topics are not sensible)
            overlapping_topics_for_this_t2 = dict()
            # add the topic from run1 to the list of topics from run1 for which t2 is the topic with maximum overlap
            if id_of_topic_with_maximum_overlap in overlapping_topics_per_t2:
                overlapping_topics_for_this_t2 = overlapping_topics_per_t2[id_of_topic_with_maximum_overlap]
            overlapping_topics_for_this_t2[t1] = maximum_overlap_size
            overlapping_topics_per_t2[id_of_topic_with_maximum_overlap] = overlapping_topics_for_this_t2



    # for each t2, get the topics t1 for which t2 had the maximum overlap, and takes the largest overlap.
    stable_topics = list()
    tid = 0
    for t2 in overlapping_topics_per_t2:
        overlapping_topics_for_this_t2 = overlapping_topics_per_t2[t2]
        (t1,overlapsize) = sorted(overlapping_topics_for_this_t2.items(), key=operator.itemgetter(1),reverse=True)[0]
        #print ("Stable topic: topic",t2,"from run2 is topic",t1,"from run1. Overlap size:",overlapsize)
        #print (overlapping_wordswithprobs_per_t1_t2[(t1,t2)])
        # Save as stable topic
        new_topic = list() # a topic is a list of two items: a tid and the wordswithprobs-list
        new_topic.append(tid)
        new_topic.append(overlapping_wordswithprobs_per_t1_t2[(t1,t2)])
        stable_topics.append(new_topic)
        tid += 1
    return stable_topics


def get_stable_topic_model_over_multiple_runs(documents,cat,num_runs,num_topics,num_words):
    corpus,dictionary = prepare_corpus(documents)
    print ("\n\nCreate initial topic model (RUN 1)")
    lda_model_init = create_topic_model(corpus,dictionary,cat,num_topics)
    topics_init = lda_model_init.show_topics(num_topics=num_topics, formatted=False, num_words=num_words)

    for run in range (2,num_runs+1):
        print ("\n\nRUN",run,"\n\n")
        lda_model= create_topic_model(corpus,dictionary,cat,num_topics)
        topics = lda_model.show_topics(num_topics=num_topics, formatted=False, num_words=num_words)
        stable_topics = find_stable_topics(topics_init,topics)
        print ("\n",len(stable_topics),"stable topics found after",run,"runs")
        for topic in stable_topics:
            print ("++",topic[0],":",topic[1])
        topics_init = stable_topics
        run += 1

    #print_topic_model(cat,num_topics,lda_model)

def create_topic_model(documents,num_topics,num_words,out):
    corpus,dictionary = prepare_corpus(documents)
    lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False, num_words=num_words)

    for topic in topics:
        print ("++",topic[0],":",topic[1])



num_topics = 50
num_words = 10
num_runs = 3
#corpus,dictionary = prepare_corpus(documents)
#create_topic_model(corpus,dictionary,"Alles",num_topics)

#get_stable_topic_model_over_multiple_runs(documents,"Alles",num_runs,num_topics,num_words)

out = open(outputfile,'w')
create_topic_model(documents,num_topics,num_words,out)
out.close()

'''
print ("\n\nPer category\n\n")
for cat in documents_per_category:
    print (">>",cat,"<<")
    num_topics = 10
    num_words = 10
    num_runs = 5
    documents = documents_per_category[cat]
    get_stable_topic_model_over_multiple_runs(documents,cat,num_runs,num_topics,num_words)
'''

out.close()