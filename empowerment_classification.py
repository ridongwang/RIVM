# pyton empowerment_classification.py /Users/suzanverberne/Data/FORUM_DATA/RIVM/Annotaties/concatenated_annotations.json forum_threads/all_KankerNL_threads.xml kankerNL_posts_labeled_automatically.tab

# In the trainset we include only items where the raters agree and the assigned value is non-empty (yes or no)
# In the testset we include also the items where the raters did not agree (value for 1 rater)

import sys
import re
#import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC

import xml.etree.ElementTree as ET
#from xlrd import open_workbook
import json
import numpy
#from matplotlib import pyplot as plt


''' use _tune = True if the script is run to compare different values for the c parameter '''
#_tune = True
_tune = False

print ("TUNE:",_tune)

annotations_filename = sys.argv[1]
corpus_path = sys.argv[2]
outfile = sys.argv[3]
author_json_file = "authors-formatted.json"

dimensions = ("narrative","emotion","factual","reflection","religious",
              "external_source","informational_support",
              "question","question_support","question_information","emotional_support")

main_dimensions = ("narrative","external_source","informational_support","question","emotional_support")
# we removed discussion_start as dimension, because it is not an empowerment construct
# we replaced factual_share by informational_support
# and support by emotional_support during the reading of the annotations file

'''
we don't include the sub dimensions in our experiments
sub_dimensions = dict()
sub_dimensions["narrative"] = ("emotion","factual","reflection","religious")
sub_dimensions["question"] = ("question_support","question_information")
'''

'''
FULL CODEBOOK:
1. Narratief - narrative
a) narratief met emoties - emotion
b) feitelijk narratief - factual
c) narrarief met reflectie op het leven - reflection
d) narratief met religie/spiritualiteit - religious

2. Verwijzing naar andere bron - external source

3. Discussie initiator - Discussion start

4. Delen van feitelijke/nuttige info voor (een) ander(en) - factual share

5. Vraag stellen - Question
a) vraag om emotionele steun - Question Support
b) vraag om informatie - Question Information

6) geven van emotionele steun - support
'''


def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëėęûüùúūôöòóõœøîïíīįìàáâäæãåçćč&@#A-ZÇĆČÉÈÊËĒĘÛÜÙÚŪÔÖÒÓŒØŌÕÎÏÍĪĮÌ0-9- \']', "", text)
    wrds = text.split()
    return wrds


def split(column,trainpercentage):
    totalitemcount = len(column)
    #print("Total no of items in column: ",totalitemcount)
    nooftrainitems = float(trainpercentage)/100*totalitemcount
    trainset = column[0:int(nooftrainitems-1)]
    testset = column[int(nooftrainitems):int(totalitemcount-1)]
    return trainset,testset


def add_value_to_target_column(target_column,dimension_name,in_trainpart):
    """
    we need this function because we treat the training data different than the test data:
    We included in the training set only the items where the raters agreed in order to avoid having conflicting training data.
    In the test set we did include the items where the raters did not agree (value for one of the two raters),
    because the quality of the classifier would be overestimated if only the agreed (clear) instances were included.
    """
    key_yes = dimension_name+"_"+dimension_name+"_yes"
    key_no = dimension_name+"_"+dimension_name+"_no"

    target_column_updated = target_column[:]
    if key_yes in annotated_item:

        if annotated_item[key_yes] == 1.0 or annotated_item[key_yes] == "2/2" or annotated_item[key_yes] == "3/3":
            # only one rater, or all raters agree on 'yes'
            target_column_updated.append("yes")
        elif not in_trainpart and annotated_item[key_yes] == "1/2" or annotated_item[key_yes] == "2/3":
            # If the raters disagree, but at least one gave the value yes, add the item to the testset
            target_column_updated.append("yes")

    elif key_no in annotated_item:
        if annotated_item[key_no] == 1.0 or annotated_item[key_no] == "2/2"or annotated_item[key_no] == "3/3":
            # only one rater, or all raters agree on 'no'
            target_column_updated.append("no")
        elif not in_trainpart and annotated_item[key_no] == "1/2" or annotated_item[key_no] == "2/3":
            # If the raters disagree, but at least one gave the value no, add the item to the testset
            target_column_updated.append("no")

    # if the two raters disagree, or if (one of the) rater(s) did not enter a value,
    # do not add the value to the target column for the trainset.

    return target_column_updated


'''MAIN'''


id_column_per_dimension = dict()
content_column_per_dimension = dict()
content_per_id = dict()
target_column_per_dimension = dict()
#categories_per_dimension = dict()
number_of_items_per_target = dict()


number_of_items = sum(1 for line in open(annotations_filename))

train_and_tunesplit = 80
tunesplit = 25
absolute_tunesplit = float(tunesplit)/100*float(train_and_tunesplit)/100*100
trainsplit = train_and_tunesplit-absolute_tunesplit
testsplit = 100-float(train_and_tunesplit)
print ("\nSplit:",train_and_tunesplit,"% for training and tuning of which",tunesplit,"% for tuning")
print("-->",trainsplit,"% for training,",absolute_tunesplit,"% for tuning, and",testsplit,"% for testing")

in_trainpart = True
itemcounter = 0
with open(annotations_filename) as annotations_file:
    for line in annotations_file:
        #print (line)
        ''' fix some inconsistensies in the output of the manual annotation '''
        line = re.sub("\"reflection_reflective_","\"reflection_reflection_",line)
        line = re.sub("\"factual_share_factual_share_","\"informational_support_informational_support_",line)
        line = re.sub("\"support_support_","\"emotional_support_emotional_support_",line)
        #print (line)

        itemcounter += 1

        ''' if we are tuning the c parameter, we train on only the train part of the train_and_tune_set '''
        splitpoint = train_and_tunesplit
        if _tune:
            splitpoint = trainsplit

        if itemcounter >= splitpoint/100*number_of_items:
            in_trainpart = False

        annotated_item = json.loads(line)
        #print(annotated_item)
        if 'token' in annotated_item and 'index' in annotated_item:
            item_id = annotated_item['index']
            content = annotated_item['token']
            content = re.sub("\?"," question_mark",content)

            #print(content)

            for dimension_name in main_dimensions:

                id_column = []
                content_column = []
                target_column = []
                if dimension_name in target_column_per_dimension:
                    id_column = id_column_per_dimension[dimension_name]
                    content_column = content_column_per_dimension[dimension_name]
                    target_column = target_column_per_dimension[dimension_name]

                target_column_updated = add_value_to_target_column(target_column,dimension_name,in_trainpart)
                if len(target_column_updated) > len(target_column):
                    """
                    if target_column_updated is the same length as target_column than no value was added
                    this happens if the value is empty, or if the two raters did not agree
                    print ("value added",len(target_column), len(target_column_updated))
                    """
                    id_column.append(item_id)
                    content_column.append(content)
                    target_column = target_column_updated

                id_column_per_dimension[dimension_name] = id_column
                content_column_per_dimension[dimension_name] = content_column
                target_column_per_dimension[dimension_name] = target_column

                '''
                if dimension_name in sub_dimensions:
                    key_yes = dimension_name+"_"+dimension_name+"_yes"
                    if key_yes in annotated_item:
                        # only classify for sub dimensions if the main dimension is yes
                        for sub_dimension_name in sub_dimensions[dimension_name]:
                            sub_target_column = []
                            id_column = []
                            content_column = []
                            if sub_dimension_name in target_column_per_dimension:
                                id_column = id_column_per_dimension[sub_dimension_name]
                                content_column = content_column_per_dimension[sub_dimension_name]
                                sub_target_column = target_column_per_dimension[sub_dimension_name]

                            sub_target_column_updated = add_value_to_target_column(sub_target_column,sub_dimension_name,in_trainpart)
                            if len(sub_target_column_updated) > len(sub_target_column):
                            # if target_column_updated is the same length as target_column than no value was added
                            # this happens if the value is empty, or if the two raters did not agree
                                id_column.append(item_id)
                                content_column.append(content)
                                sub_target_column = sub_target_column_updated

                            id_column_per_dimension[sub_dimension_name] = id_column
                            content_column_per_dimension[sub_dimension_name] = content_column
                            target_column_per_dimension[sub_dimension_name] = sub_target_column
                '''


'''
split data in train and test
'''
train_and_tuneset_per_dimension = dict()
train_and_tuneids_per_dimension = dict()
train_and_tunesetcats_per_dimension = dict()

testset_per_dimension = dict()
testids_per_dimension = dict()
testsetcats_per_dimension = dict()

for dimension in main_dimensions:
    #print ("dimension:",dimension)
    train_and_tuneset_per_dimension[dimension],testset_per_dimension[dimension] = split(content_column_per_dimension[dimension],train_and_tunesplit)
    train_and_tuneids_per_dimension[dimension],testids_per_dimension[dimension] = split(id_column_per_dimension[dimension],train_and_tunesplit)
    train_and_tunesetcats_per_dimension[dimension],testsetcats_per_dimension[dimension] = split(target_column_per_dimension[dimension],train_and_tunesplit)


'''
split training data in train and tune
'''
trainset_per_dimension = dict()
trainids_per_dimension = dict()
trainsetcats_per_dimension = dict()

tuneset_per_dimension = dict()
tuneids_per_dimension = dict()
tunesetcats_per_dimension = dict()

for dimension in main_dimensions:
    #print ("dimension:",dimension)
    trainset_per_dimension[dimension],tuneset_per_dimension[dimension] = split(train_and_tuneset_per_dimension[dimension],100-tunesplit)
    trainids_per_dimension[dimension],tuneids_per_dimension[dimension] = split(train_and_tuneids_per_dimension[dimension],100-tunesplit)
    trainsetcats_per_dimension[dimension],tunesetcats_per_dimension[dimension] = split(train_and_tunesetcats_per_dimension[dimension],100-tunesplit)

if _tune:
    testset_per_dimension = tuneset_per_dimension
    testids_per_dimension = tuneids_per_dimension
    testsetcats_per_dimension = tunesetcats_per_dimension


annotations_file.close()


for dimension_name in main_dimensions:
    print (dimension_name,"\ttrain:",len(trainids_per_dimension[dimension_name]),"\ttune:",len(tuneids_per_dimension[dimension_name]),"\ttest:",len(testids_per_dimension[dimension_name]))

    '''
    if dimension_name in sub_dimensions:
        for sub_dimension_name in sub_dimensions[dimension_name]:
            print ("   +",sub_dimension_name,"\ttrain:",len(trainids_per_dimension[dimension_name]),"\ttune:",len(tuneids_per_dimension[dimension_name]),"\ttest:",len(testids_per_dimension[dimension_name]))
    '''

print ("\ndimension\t# of yes\t# of no")
for dimension_name in main_dimensions:
    print(dimension_name,trainsetcats_per_dimension[dimension_name].count("yes")+testsetcats_per_dimension[dimension_name].count("yes"),
          trainsetcats_per_dimension[dimension_name].count("no")+testsetcats_per_dimension[dimension_name].count("no"),sep="\t")


print ("\nRead unlabeled data")
unlabeled_content_column = []
unlabeled_id_column = []
unlabeled_author_column = []
unlabeled_timestamp_column = []
posts_per_author = dict()
postlengths_per_author = dict() #key is author id, value is array of postlengths

with open (corpus_path,'r') as xml_file:
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for thread in root:
        threadid = thread.get('id')

        title = thread.find('title').text
        category = thread.find('category').text
        #print (threadid,category)

        for posts in thread.findall('posts'):
            for post in posts.findall('post'):
                postid = post.get('id')
                author = post.find('author').text
                timestamp = post.find('timestamp').text
                content = post.find('body').text
                content = re.sub("\?"," question_mark",content)
                item_id = threadid+"_"+postid
                unlabeled_content_column.append(content)
                unlabeled_id_column.append(item_id)
                unlabeled_author_column.append(author)
                unlabeled_timestamp_column.append(timestamp)
                postlength = len(tokenize(content))
                posts_for_this_author = []
                postlengths_for_this_author = []
                if author in posts_per_author:
                    posts_for_this_author = posts_per_author[author]
                    postlengths_for_this_author = postlengths_per_author[author]
                posts_for_this_author.append(item_id)
                postlengths_for_this_author.append(postlength)
                posts_per_author[author] = posts_for_this_author
                postlengths_per_author[author] = postlengths_for_this_author


number_of_contacts = dict() # key is author id
with open(author_json_file) as f:
    json_string = ""

    for line in f:
        json_string += line.rstrip()
    parsed_json = json.loads(json_string)
    for item in parsed_json:
        #print(item)
        author_id = item['id']
        contacts = item['contacts']
        number_of_contacts[author_id] = len(contacts)

print("\nNumber of occurrences in labelled data (train set)")
for dimension in main_dimensions:
    trainsetcats = trainsetcats_per_dimension[dimension]
    print(dimension,trainsetcats.count('yes'),sep="\t")


sum_precision_per_method = dict()
sum_recall_per_method = dict()
sum_f1_per_method = dict()
divide_by = dict()
classifiers_names = set()

for dimension in main_dimensions:


    #id_column = id_column_per_dimension[dimension]
    #content_column = content_column_per_dimension[dimension]
    #target_column = target_column_per_dimension[dimension]


    trainset = trainset_per_dimension[dimension]
    testset = testset_per_dimension[dimension]
    trainids = trainids_per_dimension[dimension]
    testids = testids_per_dimension[dimension]
    trainsetcats = trainsetcats_per_dimension[dimension]
    testsetcats = testsetcats_per_dimension[dimension]

    print ("train:",len(trainset),len(trainids), len(trainsetcats))
    print ("test:",len(testset),len(testids), len(testsetcats))

    if len(trainset) != len(trainsetcats):
        print ("\nERROR: columns trainset are not the same length:",len(trainset),len(trainsetcats))
        quit()
    if len(testset) != len(testsetcats):
        print ("\nERROR: columns testset are not the same length:",len(testset),len(testsetcats))
        quit()

    #print ("TRAIN:",trainids)
    #print ("TEST:",testids)

    for testid in testids:
        if testid in trainids:
            print ("\nERROR: test item in train set!",testid)
            quit()

    number_of_items_per_target_testset = dict()
    for target in testsetcats:
        if target in number_of_items_per_target_testset:
            number_of_items_per_target_testset[target] += 1
        else:
            number_of_items_per_target_testset[target] = 1

    #print ("\nNumber of items per category in the testset:")
    #print (number_of_items_per_target_testset)
    #ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), min_df=1)
    word_vectorizer = CountVectorizer(analyzer='word', min_df=1)

    vectorizernames = dict()
    #vectorizernames[ngram_vectorizer] = "character_4-grams"
    vectorizernames[word_vectorizer] = "words"

    best_f1=0
    best_recall = 0
    best_clf_vectorizer = ()
    errors_per_classifier = dict()

    for vectorizer in vectorizernames:

        Z_train = vectorizer.fit_transform(trainset)
        tfidf_transformer = TfidfTransformer()
        Z_train_tfidf = tfidf_transformer.fit_transform(Z_train)

        Z_test = vectorizer.transform(testset)
        Z_test_tfidf = tfidf_transformer.transform(Z_test)


        svm_clfs = []
        clfnames = dict()

        if _tune:
            for c in [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]:
                clf = LinearSVC(C=c).fit(Z_train_tfidf,trainsetcats)
                svm_clfs.append(clf)
                clfnames[clf] = "LinearSVC_"+str(c)
        else:
            clf = LinearSVC(C=1.0).fit(Z_train_tfidf,trainsetcats)
            svm_clfs.append(clf)
            clfnames[clf] = "LinearSVC"


        for clf in clfnames:
            if clf not in classifiers_names:
                classifiers_names.add(clfnames[clf])


        classifiers = svm_clfs

        print("category\t# of items in testset\tfeature type\tclassifier\tprecision\trecall\tF1")

        for clf in classifiers:
            errors = dict() # key is testid, value is correct target
            errors.clear()

            predicted = clf.predict(Z_test_tfidf)
            print("yes predicted:",list(predicted).count("yes"),"no predicted:",list(predicted).count("no"))
            print("yes true:",list(testsetcats).count("yes"),"no true:",list(predicted).count("no"))

            decision_values = clf.decision_function(Z_test_tfidf)

            onezerolabels = []
            for cat in testsetcats:
                if cat =="yes":
                    onezerolabels.append(1)
                else:
                    onezerolabels.append(0)
            #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(onezerolabels, decision_values)
            #plt.plot(recall, precision)
            #plt.show()


            tp = dict() # key is category
            fp = dict() # key is category
            fn = dict() # key is category
            i=0
            for testid in testids:
                assigned = predicted[i]
                correct = testsetcats[i]
                if assigned == correct:
                    if assigned in tp:
                        tp[assigned] += 1
                    else:
                        tp[assigned] = 1
                    if correct in tp:
                        tp[correct] += 1
                    else:
                        tp[correct] = 1
                else:
                    if assigned in fp:
                        fp[assigned] += 1
                    else:
                        fp[assigned] = 1
                    if correct in fn:
                        fn[correct] += 1
                    else:
                        fn[correct] = 1
                    errors[testid] = correct
                    #print ("error:",testid)
                #print (target,testid,assigned,correct)
                i += 1

            errors_per_classifier[(clfnames[clf],vectorizernames[vectorizer])] = errors

            for target in ['yes']:
                if target not in number_of_items_per_target_testset:
                    continue

                if not target in fp:
                    fp[target] = 0
                if not target in fn:
                    fn[target] = 0
                if not target in tp:
                    tp[target] = 0
                print (dimension,clfnames[clf],"tp:",tp,"fp:",fp)
                prec = 0.0
                if float(fp[target])+float(tp[target]) > 0.0:
                    prec = float(tp[target])/(float(fp[target])+float(tp[target]))
                recall = float(tp[target])/(float(tp[target])+float(fn[target]))
                f1 = 0.0
                if prec+recall > 0.0:
                    f1 = 2*(prec*recall)/(prec+recall)

                print (clfnames[clf])
                if clfnames[clf] in sum_precision_per_method:
                    sum_precision_per_method[clfnames[clf]] += prec
                    sum_recall_per_method[clfnames[clf]] += recall
                    sum_f1_per_method[clfnames[clf]] += f1
                    divide_by[clfnames[clf]] +=1
                else:
                    sum_precision_per_method[clfnames[clf]] = prec
                    sum_recall_per_method[clfnames[clf]] = recall
                    sum_f1_per_method[clfnames[clf]] = f1
                    divide_by[clfnames[clf]] = 1


                if dimension in main_dimensions:
                    print (dimension+"_"+target,"\t",number_of_items_per_target_testset[target],"\t",vectorizernames[vectorizer],"\t",clfnames[clf],"\t%.3f\t%.3f\t%.3f" % (prec,recall,f1))
                else:
                    print ("   +",dimension+"_"+target,"\t",number_of_items_per_target_testset[target],"\t",vectorizernames[vectorizer],"\t",clfnames[clf],"\t%.3f\t%.3f\t%.3f" % (prec,recall,f1))


print ("\n-------------\nOverall\n-------------")

#print (classifiers_names)
for clfname in classifiers_names:
    print("MACRO precision:",clfname,sum_precision_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO recall:",clfname,sum_recall_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO F1:",clfname,sum_f1_per_method[clfname]/divide_by[clfname],sep="\t")


print ("\nClassify unlabeled data (for each dimension)")
word_vectorizer = CountVectorizer(analyzer='word', min_df=1)
tfidf_transformer = TfidfTransformer()

posts_with_labels = dict() # key is (threadid, postid), value is array of labels
content_of_post = dict() # key is (threadid, postid), value is post content
author_of_post = dict() # key is (threadid, postid), value is post author
timestamp_of_post = dict() # key is (threadid, postid), value is post author

for dimension in main_dimensions:
    print (dimension)
    # train classifier on all labeled data (not only the 50% train split)
    labeled_id_column = trainids_per_dimension[dimension]
    labeled_content_column = trainset_per_dimension[dimension]
    labeled_target_column = trainsetcats_per_dimension[dimension]

    print ("size of training set:",len(labeled_id_column),len(labeled_content_column),len(labeled_target_column))

    Z_train = word_vectorizer.fit_transform(labeled_content_column)
    Z_train_tfidf = tfidf_transformer.fit_transform(Z_train)
    # we have to transform the unlabeled data using the same dimensions as the train data (so 'transform' instead of 'fit_transform')
    X_unlabeled = word_vectorizer.transform(unlabeled_content_column)
    X_unlabeled_tfidf = tfidf_transformer.transform(X_unlabeled)
    clf = LinearSVC().fit(Z_train_tfidf,labeled_target_column)
    predictions = clf.predict(X_unlabeled_tfidf)

    for i in range(0,len(unlabeled_id_column)-1):
        #print (dimension,unlabeled_id_column[i],predictions[i],unlabeled_content_column[i])
        (threadid,postid) = unlabeled_id_column[i].split("_")
        content_of_post[(threadid,postid)] = unlabeled_content_column[i]
        author_of_post[(threadid,postid)] = unlabeled_author_column[i]
        timestamp_of_post[(threadid,postid)] = unlabeled_timestamp_column[i]
        labels_for_this_item = []
        if (threadid,postid) in posts_with_labels:
            labels_for_this_item = posts_with_labels[(threadid,postid)]
        if predictions[i] == "yes":
            labels_for_this_item.append(dimension)
        posts_with_labels[(threadid,postid)] = labels_for_this_item


print ("\nPrint all posts with their labels")
counts_per_label = dict()
label_count_per_author = dict() # key is author id, value is dictionary with label -> count
total_number_of_assignments = 0
number_of_posts = 0
out = open(outfile,'w')
out.write("threadid\tpostid\tauthor\ttimestamp\tlabels\tcontent of post\n")
for (threadid,postid) in posts_with_labels:
    number_of_posts += 1
    author = author_of_post[(threadid,postid)]
    out.write(threadid+"\t"+postid+"\t"+author+"\t"+timestamp_of_post[(threadid,postid)]+"\t")
    labels = []
    for label in posts_with_labels[(threadid,postid)]:
        total_number_of_assignments += 1
        labels.append(label)
        if label in counts_per_label:
            counts_per_label[label] += 1
        else:
            counts_per_label[label] = 1
        label_count_for_this_author = dict()
        if author in label_count_per_author:
            label_count_for_this_author = label_count_per_author[author]
        if label in label_count_for_this_author:
            label_count_for_this_author[label] += 1
        else:
            label_count_for_this_author[label] = 1
        label_count_per_author[author] = label_count_for_this_author


    out.write(",".join(labels)+"\t"+content_of_post[(threadid,postid)]+"\n")

out.close()

print("\nPer author:")
print ("author","number_of_posts","average_postlength","number_of_contacts","\t".join(main_dimensions),sep="\t")
for author in label_count_per_author:

    #if len(posts_per_author[author]) > 30:

    labelcounts = []
    for dimension in main_dimensions:
        label_count_for_this_author = label_count_per_author[author]
        if dimension in label_count_for_this_author:
            relative_count = float(label_count_for_this_author[dimension])/float(len(posts_per_author[author]))
            #print (float(label_count_for_this_author[dimension]),float(len(posts_per_author[author])),relative_count)
            labelcounts.append(relative_count)
        else:
            labelcounts.append(0.0)
    #print (author,postlengths_per_author[author])
    mean_postlength = numpy.mean(postlengths_per_author[author])

    print (author,len(posts_per_author[author]),mean_postlength,number_of_contacts[author],"\t".join(str(x) for x in labelcounts),sep="\t")

print ("\nStats:")
print ("number of posts",number_of_posts,sep="\t")
print ("total number of assigned labels",total_number_of_assignments,sep="\t")
print ("average number of labels per post",total_number_of_assignments/number_of_posts,sep="\t")


print ("\nCounts per label:")
for label in counts_per_label:
    print (label, counts_per_label[label],sep="\t")