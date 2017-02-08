# pyton empowerment_classification.py /Users/suzanverberne/Data/FORUM_DATA/RIVM/Kanker.nl/Coding_Remco.txt

import sys
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import operator

posts_with_targets_file = sys.argv[1]

def tokenize(t):
    text = t.lower()
    text = re.sub("\n"," ",text)
    text = re.sub(r'<[^>]+>',"",text) # remove all html markup
    text = re.sub('[^a-zèéeêëûüùôöòóœøîïíàáâäæãåA-Z0-9- \']', "", text)
    wrds = text.split()
    return wrds


def fix_spelling_errors(t):
    t = re.sub("[.,]","",t)
    t = re.sub("^ ","",t)
    t = re.sub(" $","",t)
    t = re.sub("  "," ",t)
    t = re.sub("informaiton","information",t)
    t = re.sub("infromation","information",t)
    t = re.sub("disclossure","disclosure",t)
    t = re.sub("durration","duration",t)
    t = re.sub("infomrational","informational",t)
    t = re.sub("interpersona/","interpersonal/",t)
    t = re.sub("faction","factual",t)
    t = re.sub("initator","initiator",t)
    t = re.sub("intiator","initiator",t)
    t = re.sub("upon ","on ",t)
    t = re.sub("treatment-specific","treatment-related",t)
    t = re.sub("reflection","reflection on ones life",t)
    t = re.sub("^information$","informational",t)
    t = re.sub("^informational$","informational support",t)
    t = re.sub("^emotional$","emotional support",t)
    return t

def split(column,trainpercentage):
    totalitemcount = len(column)
    #print("Total no of items in column: ",totalitemcount)
    nooftrainitems = float(trainpercentage)/100*totalitemcount
    trainset = column[0:int(nooftrainitems-1)]
    testset = column[int(nooftrainitems):int(totalitemcount-1)]
    return trainset,testset

'''MAIN'''


id_column = []
target_column = []
content_column = []
content_per_id = dict()
categories = []
number_of_items_per_target = dict()

i = 0
with open(posts_with_targets_file,'r') as posts_with_targets:
    for line in posts_with_targets:
        columns = line.rstrip().split('\t')
        if 'threadid' in columns[0]:
            continue
        threadid,postid,author,timestamp,body,upvotes = columns[0:6]
        if len(columns) > 6:
            targets = columns[6].lower().split(", ")
            item_id = threadid+"_"+postid
            for target in targets:
                target = fix_spelling_errors(target)
                id_column.append(item_id)
                target_column.append(target)
                #print (i,id_column[i],target_column[i])
                i += 1
                if target not in categories:
                    #print (target)
                    categories.append(target)
                    number_of_items_per_target[target] = 1
                else:
                    number_of_items_per_target[target] += 1
                #content = author+" "+body
                content = body
                content_column.append(content)
                content_per_id[item_id] = content


posts_with_targets.close()

noofitems = len(id_column)
noofcats = len(categories)
if len(id_column) != len(content_column) != len(target_column):
    print ("\nERROR: columns are not the same length:",len(id_column),len(content_column), len(target_column))
    quit()

print("Total no of items in dataset: ",noofitems)
print("Total no of categories in dataset: ",noofcats)

trainsplit = 50
print ("Split:",trainsplit,"% training and remainder for testing")
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

print ("\nNumber of items per category in the testset:")
print (number_of_items_per_target_testset)
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), min_df=1)
word_vectorizer = CountVectorizer(analyzer='word', min_df=1)

vectorizernames = dict()
vectorizernames[ngram_vectorizer] = "character_4-grams"
vectorizernames[word_vectorizer] = "words"

best_f1=0
best_recall = 0
best_clf_vectorizer = ()
errors_per_classifier = dict()

for vectorizer in vectorizernames:

    print ("Feature type:",vectorizernames[vectorizer])
    Z_train = vectorizer.fit_transform(trainset)
    #print ("Feature names:",vectorizer.get_feature_names())
    print ("No of features:",len(vectorizer.get_feature_names()))

    #tfidf_transformer = TfidfTransformer(use_idf=False)
    tfidf_transformer = TfidfTransformer()
    Z_train_tfidf = tfidf_transformer.fit_transform(Z_train)
    print("Train dimensions:",Z_train_tfidf.shape)
    #matrix = X.toarray()
    Z_test = vectorizer.transform(testset)
    Z_test_tfidf = tfidf_transformer.transform(Z_test)
    print("Test dimensions:",Z_test_tfidf.shape)
    #print ("Test set tfidf matrix:",Z_test_tfidf)

    #print (categories)

    clfnames = dict()

#    clf1 = MultinomialNB().fit(Z_train_tfidf,trainsetcats)
    clf2 = LinearSVC().fit(Z_train_tfidf,trainsetcats)
#    clf3 = Perceptron().fit(Z_train_tfidf,trainsetcats)
#    clf4 = RandomForestClassifier().fit(Z_train_tfidf,trainsetcats)
#    clf5 = KNeighborsClassifier().fit(Z_train_tfidf,trainsetcats)
    clf6 = LogisticRegression(multi_class='ovr').fit(Z_train_tfidf,trainsetcats) # one-versus-all
    #print("Params:", clf6.get_params())
    #print("Intercept:",clf6.intercept_)
    #print("Coefficients:",clf6.coef_[0])


#    clfnames[clf1] = "MultinomialNB"
    clfnames[clf2] = "LinearSVC"
#    clfnames[clf3] = "Perceptron"
#    clfnames[clf4] = "RandomForestClassifier"
#    clfnames[clf5] = "KNeighborsClassifier"
    clfnames[clf6] = "LogisticRegression"

    print ("\nMost important features according to LogisticRegression model:")
    c =0
    for target in categories:
        if target not in number_of_items_per_target_testset:
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

        for target in categories:
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

            print (target,"\t",number_of_items_per_target_testset[target],"\t",vectorizernames[vectorizer],"\t",clfnames[clf],"\t%.3f\t%.3f\t%.3f" % (prec,recall,f1))



