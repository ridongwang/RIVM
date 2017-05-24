# pyton empowerment_classification.py /Users/suzanverberne/Data/FORUM_DATA/RIVM/Annotaties/concatenated_annotations.json

import sys
import re
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import operator
from xlrd import open_workbook
import json


annotations_filename = sys.argv[1]

dimensions = ("discussion_start","emotion","external_source","factual","factual_share","narrative",
              "question","question_information","question_support","reflection","religious","support")

main_dimensions = ("narrative","external_source","discussion_start","factual","question","support")


'''
annotations_dir = sys.argv[1]

xlsx_files = []
for file in os.listdir(annotations_dir):
    if ".xlsx" in file and not "~" in file:
        xlsx_files.append(annotations_dir+"/"+file)
'''



def split(column,trainpercentage):
    totalitemcount = len(column)
    #print("Total no of items in column: ",totalitemcount)
    nooftrainitems = float(trainpercentage)/100*totalitemcount
    trainset = column[0:int(nooftrainitems-1)]
    testset = column[int(nooftrainitems):int(totalitemcount-1)]
    return trainset,testset

'''MAIN'''


id_column = []
content_column = []
content_per_id = dict()
target_column_per_dimension = dict()
#categories_per_dimension = dict()
number_of_items_per_target = dict()


with open(annotations_filename) as annotations_file:
    for line in annotations_file:
        annotated_item = json.loads(line)
        #print(annotated_item)
        if 'token' in annotated_item and 'index' in annotated_item:
            item_id = annotated_item['index']
            content = annotated_item['token']
            content = re.sub("\?","question_mark",content)

            #print(content)
            id_column.append(item_id)
            content_column.append(content)
            for dimension_name in main_dimensions:
                target_column = []
                if dimension_name in target_column_per_dimension:
                    target_column = target_column_per_dimension[dimension_name]


                key_yes = dimension_name+"_"+dimension_name+"_yes"
                key_no = dimension_name+"_"+dimension_name+"_no"

                if key_yes in annotated_item:

                    if annotated_item[key_yes] == 1.0 or annotated_item[key_yes] == "2/2" or annotated_item[key_yes] == "3/3":
                        target_column.append("yes")
                    else:
                        target_column.append("?")
                        #print (item_id,key_yes,annotated_item[key_yes])
                elif key_no in annotated_item:
                    #print (key_no,annotated_item[key_no])
                    if annotated_item[key_no] == 1.0 or annotated_item[key_no] == "2/2"or annotated_item[key_no] == "3/3":
                        target_column.append("no")
                    else:
                        target_column.append("?")
                        #print (item_id,key_no,annotated_item[key_no])
                else:
                    target_column.append("?")

                target_column_per_dimension[dimension_name] = target_column

print("number of items:",len(id_column),len(content_column))
for dimension_name in main_dimensions:
    print (dimension_name,len(target_column_per_dimension[dimension_name]),target_column_per_dimension[dimension_name])




'''
for filename in xlsx_files:
    print(filename)
    book = open_workbook(filename,encoding_override='utf-8')
    sheet = book.sheet_by_index(0)

    # read header values into the list
    keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]

'''



if len(id_column) != len(content_column):
    print ("\nERROR: columns are not the same length:",len(id_column),len(content_column), len(target_column))
    quit()


trainsplit = 50
print ("\nSplit:",trainsplit,"% training and remainder for testing")
sum_precision_per_method = dict()
sum_recall_per_method = dict()
sum_f1_per_method = dict()
divide_by = dict()
classifiers_names = set()

for dimension in main_dimensions:
    print ("\n-------------\n",dimension,"\n-------------")
    target_column = target_column_per_dimension[dimension]

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
        c=0
        for target in clf6.classes_:
            if target =="yes":
                coefficients = clf6.coef_[c] #get the coefficients for the c's target
                #print("No of coefficients:",len(coefficients))

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
            c += 1


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



                print (dimension+"_"+target,"\t",number_of_items_per_target_testset[target],"\t",vectorizernames[vectorizer],"\t",clfnames[clf],"\t%.3f\t%.3f\t%.3f" % (prec,recall,f1))

print ("\n-------------\nOverall\n-------------")

#print (classifiers_names)
for clfname in classifiers_names:
    print("MACRO precision:",clfname,sum_precision_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO recall:",clfname,sum_recall_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO F1:",clfname,sum_f1_per_method[clfname]/divide_by[clfname],sep="\t")



