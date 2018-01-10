# python longitudinal.py per_author kankerNL_posts_labeled_automatically.tab

import sys
import os
import re
#from datetime import datetime
import numpy as np
from dateutil.parser import parse
import operator


per_author_dir = sys.argv[1]
labelled_corpus = sys.argv[2]


nonmods = ("e48f6120-933c-0132-71d9-2a0161707897", " da4a6c00-8a5f-0130-db36-5a0007c06fd5", " cacee970-8f3c-0130-dcd2-5a0007c06fd5", " f16d33c0-884e-0132-6ecb-2a0161707897", " 36b0e8f0-89bb-0130-daac-5a0007c06fd5", " 3dac0720-5d07-0132-6144-2a0161707897")


labels_per_post = dict()
timestamp_per_post = dict()

def split_items_in_bins (all_postlabels):
    bins = dict()
    items_per_bin = len(all_postlabels)//no_of_bins
    #print (author,"items per bin:",items_per_bin,author_postlabels)
    start=0
    partition=0
    for partition in range(1,no_of_bins+1):
        list_of_items_in_bin = list()
        #print ("bin", partition)
        i=0
        for i in range(start,start+items_per_bin):
            postlabels = all_postlabels[i]
            if partition in bins:
                list_of_items_in_bin = bins[partition]
            list_of_items_in_bin.append(postlabels)
            #print (i, postlabels)
            bins[partition] = list_of_items_in_bin
        start = i+1

    '''
    list_of_items_in_last_bin = bins[partition]
    for i in range(start,len(author_postlabels)):
        postlabels = author_postlabels[i]
        list_of_items_in_last_bin.append(postlabels)
        #print (i,postlabels)
        bins[partition] = list_of_items_in_last_bin
    '''
    return bins

def count_labels_per_bin (labels_per_partition):
    count_per_label_per_bin = dict()
    for partition in labels_per_partition:
        #print (partition,labels_per_partition_over_all_authors[partition])
        #print (partition)
        count_per_label = dict()
        for label in labels_per_partition[partition]:
            if label in count_per_label:
                count_per_label[label] += 1
            else:
                count_per_label[label] = 1
        #for label in count_per_label:
        #    print("\t",label,"\t",count_per_label[label])
        count_per_label_per_bin[partition] = count_per_label
    return count_per_label_per_bin


with open(labelled_corpus,'r',encoding='utf-8') as labelfile:
    for line in labelfile:
        #print(line.rstrip())
        columns = line.rstrip().split("\t")
        if len(columns) > 4:
            itemid,postid,author,timestamp,labels,content = columns
            if postid != "postid":
                labels_per_post[(itemid,postid)] = labels
                #print (itemid,postid)
                #print(timestamp)
                posttime = parse(timestamp)
                timestamp_per_post[(itemid,postid)] = posttime

labelfile.close()

time_ordered_postlabels_alldata = []

sorted_postids = sorted(timestamp_per_post.items(), key=operator.itemgetter(1))
print (sorted_postids)
for (post,timestamp) in sorted_postids:
    (itemid,postid) = post
    #print (itemid,postid,timestamp_per_post[(itemid,postid)])
    time_ordered_postlabels_alldata.append(labels_per_post[itemid,postid])

#print (time_ordered_postlabels_alldata)


time_ordered_postlabels_per_author = dict()
timestamps_per_author = dict()

for file in os.listdir(per_author_dir):
    #print(file)
    if '.tsv' in file and not '.zip' in file:
        with open(per_author_dir+"/"+file,'r',encoding='utf-8') as tsvfile:
            for line in tsvfile:
                #print (line.rstrip())
                itemid,postid,author,timestamp,body,ups = line.rstrip().split("\t")
                posttime = parse(timestamp)
                if (itemid,postid) in labels_per_post:
                    #print (itemid,postid,author,labels_per_post[(itemid,postid)])

                    author_postlabels = []
                    author_timestamps = []
                    if author in time_ordered_postlabels_per_author:
                        author_postlabels = time_ordered_postlabels_per_author[author]
                        author_timestamps = timestamps_per_author[author]
                    author_postlabels.append(labels_per_post[(itemid,postid)])
                    author_timestamps.append(posttime)
                    time_ordered_postlabels_per_author[author] = author_postlabels
                    timestamps_per_author[author] = author_timestamps

                else:
                    print ("not in labelled data:",(itemid,postid))
        tsvfile.close()

active_time_periods = []
for author in timestamps_per_author:
    author_timestamps = timestamps_per_author[author]
    first = author_timestamps[0]
    last = author_timestamps[-1]
    tdelta = last - first
    print (author,tdelta.days)
    active_time_periods.append(tdelta.days)

print("mean:",np.mean(active_time_periods),"stdev:",np.std(active_time_periods))

### FOR THE AUTHORS WITH > 30 POSTS ###

no_of_bins = 5
labels_per_partition_over_all_authors = dict() #key is partition, value is array of labels
labels_per_partition_over_nonmods = dict() #key is partition, value is array of labels
for author in time_ordered_postlabels_per_author:

    author_postlabels = time_ordered_postlabels_per_author[author]
    bins = split_items_in_bins(author_postlabels)

    # AGGREGATE OVER AUTHORS:
    for partition in bins:
        items = bins[partition]
        labels_for_partition = []
        labels_for_partition_over_all_authors = []
        labels_for_partition_over_nonmods = []
        if partition in labels_per_partition_over_all_authors:
            labels_for_partition_over_all_authors = labels_per_partition_over_all_authors[partition]
        if partition in labels_per_partition_over_nonmods:
            labels_for_partition_over_nonmods = labels_per_partition_over_nonmods[partition]
        for item in items:
            labels = item.split(",")
            for label in labels:
                labels_for_partition.append(label)
                labels_for_partition_over_all_authors.append(label)
                if author in nonmods:
                    labels_for_partition_over_nonmods.append(label)
        labels_per_partition_over_all_authors[partition] = labels_for_partition_over_all_authors
        labels_per_partition_over_nonmods[partition] = labels_for_partition_over_nonmods
        #print (partition,labels_for_partition)

count_per_label_per_bin_authors = count_labels_per_bin(labels_per_partition_over_all_authors)
count_per_label_per_bin_nonmods = count_labels_per_bin(labels_per_partition_over_nonmods)


print("\nDevelopment over time for the authors with more than 30 posts:")
labels = ("narrative","external_source","informational_support","question","emotional_support")
for label in labels:
    print ("\t",label,end='',flush=True)
print()
for partition in count_per_label_per_bin_authors:

    print (partition,end='',flush=True)
    labelcount = count_per_label_per_bin_authors[partition]
    for label in labels:
        print ("\t",labelcount[label],end='',flush=True)
    print()


#### FOR ALL DATA ####

labels_per_partition_alldata = split_items_in_bins(time_ordered_postlabels_alldata)
#for partition in bins:
#    print (partition,bins[partition])
count_per_label_per_bin_alldata = count_labels_per_bin(labels_per_partition_alldata)

print("\nDevelopment over time for the complete corpus:")
for label in labels:
    print ("\t",label,end='',flush=True)
print()
for partition in count_per_label_per_bin_alldata:

    print (partition,end='',flush=True)
    labelcount = count_per_label_per_bin_alldata[partition]
    for label in labels:
        print ("\t",labelcount[label],end='',flush=True)
    print()


#### FOR NON-MODERATORS WITH > 30 POSTS ###

print("\nDevelopment over time for the non-moderator authors with more than 30 posts:")
labels = ("narrative","external_source","informational_support","question","emotional_support")
for label in labels:
    print ("\t",label,end='',flush=True)
print()
for partition in count_per_label_per_bin_nonmods:

    print (partition,end='',flush=True)
    labelcount = count_per_label_per_bin_nonmods[partition]
    for label in labels:
        count = 0
        if label in labelcount:
            count = labelcount[label]
        print ("\t",count,end='',flush=True)
    print()
