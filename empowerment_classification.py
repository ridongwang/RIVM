# pyton empowerment_classification.py /Users/suzanverberne/Data/FORUM_DATA/RIVM/Annotaties/concatenated_annotations.json forum_threads/all_KankerNL_threads.xml kankerNL_posts_labeled_automatically.tab

import sys
import re
#import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import operator
import xml.etree.ElementTree as ET
#from xlrd import open_workbook
import json


annotations_filename = sys.argv[1]
corpus_path = sys.argv[2]
outfile = sys.argv[3]

dimensions = ("narrative","emotion","factual","reflection","religious",
              "external_source","discussion_start","factual_share",
              "question","question_support","question_information","support")

main_dimensions = ("narrative","external_source","discussion_start","factual_share","question","support")

sub_dimensions = dict()
sub_dimensions["narrative"] = ("emotion","factual","reflection","religious")
sub_dimensions["question"] = ("question_support","question_information")

'''
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




def split(column,trainpercentage):
    totalitemcount = len(column)
    #print("Total no of items in column: ",totalitemcount)
    nooftrainitems = float(trainpercentage)/100*totalitemcount
    trainset = column[0:int(nooftrainitems-1)]
    testset = column[int(nooftrainitems):int(totalitemcount-1)]
    return trainset,testset


def add_value_to_target_column(target_column,dimension_name):
    key_yes = dimension_name+"_"+dimension_name+"_yes"
    key_no = dimension_name+"_"+dimension_name+"_no"
    target_column_updated = []
    #for item in target_column:
    #    target_column_updated.append(item)
    target_column_updated = target_column[:]
    if key_yes in annotated_item:

        if annotated_item[key_yes] == 1.0 or annotated_item[key_yes] == "2/2" or annotated_item[key_yes] == "3/3":
            target_column_updated.append("yes")
        #else:
            #target_column.append("?")
            #print (item_id,key_yes,annotated_item[key_yes])
    elif key_no in annotated_item:
        #print (key_no,annotated_item[key_no])
        if annotated_item[key_no] == 1.0 or annotated_item[key_no] == "2/2"or annotated_item[key_no] == "3/3":
            target_column_updated.append("no")
        #else:
            #target_column.append("?")
            #print (item_id,key_no,annotated_item[key_no])
    #else:
        #target_column.append("?")
    return target_column_updated

'''MAIN'''





id_column_per_dimension = dict()
content_column_per_dimension = dict()
content_per_id = dict()
target_column_per_dimension = dict()
#categories_per_dimension = dict()
number_of_items_per_target = dict()


with open(annotations_filename) as annotations_file:
    for line in annotations_file:
        line = re.sub("reflection_reflective","reflection_reflection",line)
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


                target_column_updated = add_value_to_target_column(target_column,dimension_name)
                if len(target_column_updated) > len(target_column):
                    # if target_column_updated is the same length as target_column than no value was added
                    # this happens if the value is empty, or if the two raters did not agree
                    #print ("value added",len(target_column), len(target_column_updated))
                    id_column.append(item_id)
                    content_column.append(content)
                    target_column = target_column_updated

                id_column_per_dimension[dimension_name] = id_column
                content_column_per_dimension[dimension_name] = content_column
                target_column_per_dimension[dimension_name] = target_column

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

                            sub_target_column_updated = add_value_to_target_column(sub_target_column,sub_dimension_name)
                            if len(sub_target_column_updated) > len(sub_target_column):
                            # if target_column_updated is the same length as target_column than no value was added
                            # this happens if the value is empty, or if the two raters did not agree
                                id_column.append(item_id)
                                content_column.append(content)
                                sub_target_column = sub_target_column_updated

                            id_column_per_dimension[sub_dimension_name] = id_column
                            content_column_per_dimension[sub_dimension_name] = content_column
                            target_column_per_dimension[sub_dimension_name] = sub_target_column



annotations_file.close()
for dimension_name in main_dimensions:
    print (dimension_name,"\t",len(id_column_per_dimension[dimension_name]))#len(target_column_per_dimension[dimension_name]),target_column_per_dimension[dimension_name])

    if dimension_name in sub_dimensions:
        for sub_dimension_name in sub_dimensions[dimension_name]:
            print ("   +",sub_dimension_name,"\t",len(id_column_per_dimension[sub_dimension_name]))#,len(target_column_per_dimension[sub_dimension_name]),target_column_per_dimension[sub_dimension_name])

print ("Read unlabeled data")
unlabeled_content_column = []
unlabeled_id_column = []
unlabeled_author_column = []
unlabeled_timestamp_column = []
posts_per_author = dict()

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
                posts_for_this_author = []
                if author in posts_per_author:
                    posts_for_this_author = posts_per_author[author]
                posts_for_this_author.append(item_id)
                posts_per_author[author] = posts_for_this_author

'''
for filename in xlsx_files:
    print(filename)
    book = open_workbook(filename,encoding_override='utf-8')
    sheet = book.sheet_by_index(0)

    # read header values into the list
    keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]

'''






trainsplit = 50
print ("\nSplit:",trainsplit,"% training and remainder for testing")
sum_precision_per_method = dict()
sum_recall_per_method = dict()
sum_f1_per_method = dict()
divide_by = dict()
classifiers_names = set()

for dimension in dimensions:
    if dimension in main_dimensions:
        print ("\n-------------\nMAIN DIMENSION:",dimension,"\n-------------")
    else:
        print ("\n-------------\nSUB DIMENSION:",dimension,"\n-------------")

    id_column = id_column_per_dimension[dimension]
    content_column = content_column_per_dimension[dimension]
    target_column = target_column_per_dimension[dimension]

    print (len(id_column),len(content_column), len(target_column))

    if len(id_column) != len(target_column):
        print ("\nERROR: columns are not the same length:",len(id_column),len(content_column), len(target_column))
        quit()

    (trainset,testset) = split(content_column,trainsplit)
    (trainids,testids) = split(id_column,trainsplit)
    (trainsetcats,testsetcats) = split(target_column,trainsplit)

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

        #print ("Feature type:",vectorizernames[vectorizer])
        Z_train = vectorizer.fit_transform(trainset)
        #print ("Feature names:",vectorizer.get_feature_names())
        #print ("No of features:",len(vectorizer.get_feature_names()))

        #tfidf_transformer = TfidfTransformer(use_idf=False)
        tfidf_transformer = TfidfTransformer()
        Z_train_tfidf = tfidf_transformer.fit_transform(Z_train)
        #print("Train dimensions:",Z_train_tfidf.shape)
        #matrix = X.toarray()
        Z_test = vectorizer.transform(testset)
        Z_test_tfidf = tfidf_transformer.transform(Z_test)
        #print("Test dimensions:",Z_test_tfidf.shape)
        #print ("Test set tfidf matrix:",Z_test_tfidf)

        #print (categories)



    #    clf1 = MultinomialNB().fit(Z_train_tfidf,trainsetcats)
        clf2 = LinearSVC().fit(Z_train_tfidf,trainsetcats)
    #    clf3 = Perceptron().fit(Z_train_tfidf,trainsetcats)
    #    clf4 = RandomForestClassifier().fit(Z_train_tfidf,trainsetcats)
    #    clf5 = KNeighborsClassifier().fit(Z_train_tfidf,trainsetcats)
        clf6 = LogisticRegression(multi_class='ovr').fit(Z_train_tfidf,trainsetcats) # one-versus-all
        #print("Params:", clf6.get_params())
        #print("Intercept:",clf6.intercept_)
        #print("Coefficients:",clf6.coef_[0])

        clfnames = dict()

    #    clfnames[clf1] = "MultinomialNB"
        clfnames[clf2] = "LinearSVC"
    #    clfnames[clf3] = "Perceptron"
    #    clfnames[clf4] = "RandomForestClassifier"
    #    clfnames[clf5] = "KNeighborsClassifier"
        clfnames[clf6] = "LogisticRegression"
        for clf in clfnames:
            if clf not in classifiers_names:
                classifiers_names.add(clfnames[clf])

        print ("\nMost important features according to LogisticRegression model:")
        '''
        c=0
        print(dimension,clf6.classes_)
        for target in clf6.classes_:
            print (c,target,clf6.coef_[c])
            if target =="yes":
                print ("yes!",c,target,clf6.classes_[c])

                coefficients = clf6.coef_[c] #get the coefficients for the c's target
                '''
        coefficients = clf6.coef_[0]
        print("No of coefficients:",len(coefficients))

        feats_with_coefs = dict()
        k=0
        for featname in vectorizer.get_feature_names():
            coef = coefficients[k]
            feats_with_coefs[featname] = coef
            k += 1
        sorted_feats = sorted(feats_with_coefs.items(), key=operator.itemgetter(1), reverse=True)


        print ("Top",vectorizernames[vectorizer],"for '",dimension,"'")
        for top in range(0,10):
            print (sorted_feats[top])

            #c += 1


    #    classifiers = [clf1,clf2,clf3,clf4,clf5,clf6]
        classifiers = [clf2,clf6]

        print("category\t# of items in testset\tfeature type\tclassifier\tprecision\trecall\tF1")

        for clf in classifiers:
            errors = dict() # key is testid, value is correct target
            errors.clear()
            #out = open("classification."+vectorizernames[vectorizer]+"."+clfnames[clf]+".txt",'w')

            predicted = clf.predict(Z_test_tfidf)
            #print(predicted)
            #print(testsetcats)

            tp = 0
            fp = dict() # key is category
            fn = dict() # key is category
            i=0
            for testid in testids:
                assigned = predicted[i]
                correct = testsetcats[i]
                #print(testid,correct,assigned)
                if assigned == correct:
                    tp += 1
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
            #out.close()
            #print ("TP:",tp)
            errors_per_classifier[(clfnames[clf],vectorizernames[vectorizer])] = errors

            for target in ('yes','no'):
                if target not in number_of_items_per_target_testset:
                    continue
                if number_of_items_per_target_testset[target] < 10:
                    continue
                if not target in fp:
                    fp[target] = 0
                if not target in fn:
                    fn[target] = 0
                prec = float(tp)/(float(fp[target])+float(tp))
                recall = float(tp)/(float(tp)+float(fn[target]))
                f1 = 2*(prec*recall)/(prec+recall)

                #print (clfnames[clf])
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


print ("\nTrain classifier on all labeled data and classify unlabeled data (for each dimension)")
word_vectorizer = CountVectorizer(analyzer='word', min_df=1)
tfidf_transformer = TfidfTransformer()

posts_with_labels = dict() # key is (threadid, postid), value is array of labels
content_of_post = dict() # key is (threadid, postid), value is post content
author_of_post = dict() # key is (threadid, postid), value is post author
timestamp_of_post = dict() # key is (threadid, postid), value is post author

for dimension in main_dimensions:
    print (dimension)
    # train classifier on all labeled data (not only the 50% train split)
    labeled_id_column = id_column_per_dimension[dimension]
    labeled_content_column = content_column_per_dimension[dimension]
    labeled_target_column = target_column_per_dimension[dimension]
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
print ("author","number_of_posts","\t".join(main_dimensions),sep="\t")
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

    print (author,len(posts_per_author[author]),"\t".join(str(x) for x in labelcounts),sep="\t")

print ("\nStats:")
print ("number of posts",number_of_posts,sep="\t")
print ("total number of assigned labels",total_number_of_assignments,sep="\t")
print ("average number of labels per post",total_number_of_assignments/number_of_posts,sep="\t")


print ("\nCounts per label:")
for label in counts_per_label:
    print (label, counts_per_label[label],sep="\t")