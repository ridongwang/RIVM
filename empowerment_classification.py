# pyton empowerment_classification.py Directory-with-xlsx-files/

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

annotations_dir = sys.argv[1]
annotations_files = []
for file in os.listdir(annotations_dir):
    if ".xlsx" in file and not "~" in file:
        annotations_files.append(annotations_dir+"/"+file)




def split(column,trainpercentage):
    totalitemcount = len(column)
    #print("Total no of items in column: ",totalitemcount)
    nooftrainitems = float(trainpercentage)/100*totalitemcount
    trainset = column[0:int(nooftrainitems-1)]
    testset = column[int(nooftrainitems):int(totalitemcount-1)]
    return trainset,testset

'''MAIN'''


id_column = []
target_column_per_dimension = dict()
categories_per_dimension = dict()

content_column = []
content_per_id = dict()
number_of_items_per_target = dict()

for filename in annotations_files:
    print(filename)
    book = open_workbook(filename,encoding_override='utf-8')
    sheet = book.sheet_by_index(0)

    # read header values into the list
    keys = [sheet.cell(0, col_index).value for col_index in range(sheet.ncols)]
    dimensions = keys[11:]

    #dict_list = []
    for row_index in range(1, sheet.nrows):
        #index	appreciation_count	category	userid	token	thread_id	level	created_at	title	datetime	author	factual_share	emotion	question_information	question	question_support	narrative	discussion_start	reflection	factual	external_source	support	religious
        item_id = sheet.cell(row_index, 0).value
        if item_id in id_column:
            print ("item is already in the data (by another annotator). Do not include twice")
        else:
            content = sheet.cell(row_index,4).value
            content = re.sub("\?"," question_mark",content)
            id_column.append(item_id)
            content_column.append(content) #  classify on content only
            content_per_id[item_id] = content

            i = 11
            for dimension in dimensions:
                target = content = sheet.cell(row_index,i).value
                if target == '':
                    target = 'UNDEFINED'

                categories = set()
                target_column = []
                if dimension in categories_per_dimension:
                    categories = categories_per_dimension[dimension]
                    target_column = target_column_per_dimension[dimension]
                target_column.append(target)
                if target not in categories:
                    categories.add(target)
                if target not in number_of_items_per_target:
                    number_of_items_per_target[target] = 1
                else:
                    number_of_items_per_target[target] += 1
                target_column_per_dimension[dimension] = target_column
                categories_per_dimension[dimension] = categories


                i += 1


print ("categories per dimension:",categories_per_dimension)
print (len(id_column),id_column)
print (len(content_column),content_column)
#for dimension in target_column_per_dimension:
#    print (dimension,len(target_column_per_dimension[dimension]),sep="\t")
for target in number_of_items_per_target:
    print (target,number_of_items_per_target[target],sep="\t")


if len(id_column) != len(content_column):
    print ("\nERROR: columns are not the same length:",len(id_column),len(content_column), len(target_column))
    quit()
noofitems = len(id_column)
noofdimensions = len(categories_per_dimension)


print("Total no of items in dataset: ",noofitems,sep="\t")
print("No of classification dimensions in dataset: ",noofdimensions,sep="\t")

trainsplit = 50
print ("Split:",trainsplit,"% training and remainder for testing")
sum_precision_per_method = dict()
sum_recall_per_method = dict()
sum_f1_per_method = dict()
divide_by = dict()
classifiers_names = set()

for dimension in categories_per_dimension:
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
        c =0
        for target in categories_per_dimension[dimension]:
            if target not in number_of_items_per_target_testset or '_yes' not in target:
                continue
            if number_of_items_per_target_testset[target] < 10:
                continue
            coefficients = clf6.coef_[c] #get the coefficients for the c's target
            #print("No of coefficients:",len(coefficients))

            feats_with_coefs = dict()
            k=0
            for featname in vectorizer.get_feature_names():
                coef = coefficients[k]
                feats_with_coefs[featname] = coef
                k += 1
            sorted_feats = sorted(feats_with_coefs.items(), key=operator.itemgetter(1), reverse=True)


            print ("Top",vectorizernames[vectorizer],"for target \'",target,"\':")
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

            for target in categories_per_dimension[dimension]:
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



                print (target,"\t",number_of_items_per_target_testset[target],"\t",vectorizernames[vectorizer],"\t",clfnames[clf],"\t%.3f\t%.3f\t%.3f" % (prec,recall,f1))

print ("\n-------------\nOverall\n-------------")

#print (classifiers_names)
for clfname in classifiers_names:
    print("MACRO precision:",clfname,sum_precision_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO recall:",clfname,sum_recall_per_method[clfname]/divide_by[clfname],sep="\t")
    print("MACRO F1:",clfname,sum_f1_per_method[clfname]/divide_by[clfname],sep="\t")