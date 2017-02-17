# python longitudinal.py forum_threads/all_KankerNL_threads.xml per_author

import sys
import re
import xml.etree.ElementTree as ET
from dateutil.parser import parse
import operator


threadsfile = sys.argv[1]
outdir = sys.argv[2]



class Post:

    def __init__(self,postid,author,timestamp,body,parentid,ups,downs):
        self.postid = postid
        self.author = author
        self.timestamp = timestamp
        self.body = body
        self.parentid = parentid
        self.ups = ups
        self.downs = downs

    def printXML(self,out):
        out.write("<post id=\""+self.postid+"\">\n<author>"+self.author+"</author>\n<timestamp>"+str(self.timestamp)+"</timestamp>\n<parentid>"+self.parentid+"</parentid>\n<body>"+self.body+"</body>\n<upvotes>"+str(self.ups)+"</upvotes>\n<downvotes>"+str(self.downs)+"</downvotes>\n</post>\n")

    def printTXT(self,out):
        out.write(self.body+"\n")

    def printTSV(self,threadid,out):
        out.write(threadid+"\t"+self.postid+"\t"+self.author+"\t"+self.timestamp+"\t"+self.body+"\t"+str(self.ups)+"\n")


posts_per_author = dict() #key is author name, value is set of tuples threadid,postid
post_objects = dict() # key is (threadid,postid), value is post object
timestamp_per_postid = dict() # key is (threadid,postid), value is parsed timestamp

with open (threadsfile,'r') as xml_file:
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
                ups = post.find('upvotes').text
                parentid = post.find('parentid').text
                posts_for_this_author = set()
                if author in posts_per_author:
                    posts_for_this_author = posts_per_author[author]
                posts_for_this_author.add((threadid,postid))
                posts_per_author[author] = posts_for_this_author
                post_object = Post(postid,author,timestamp,content,parentid,ups,0)
                dt = parse(timestamp)
                timestamp_per_postid[(threadid,postid)] = dt
                post_objects[(threadid,postid)] = post_object


posts_sorted_by_timestamp = sorted(timestamp_per_postid.items(), key=operator.itemgetter(1))


for author in posts_per_author:
    posts_for_this_author = posts_per_author[author]
    print(author,len(posts_for_this_author))
    if len(posts_for_this_author) > 30:
        out = open(outdir+"/"+author+".tsv",'w')
        for ((threadid,postid),timestamp) in posts_sorted_by_timestamp:
            print (threadid,postid,timestamp)
            if (threadid,postid) in posts_for_this_author:
                post = post_objects[(threadid,postid)]
                post.printTSV(threadid,out)
        out.close()




