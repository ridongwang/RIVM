# coding=utf-8
# python json2xml_iknl.py discussions-formatted.json threads_dir KankerNL_posts_formatRemco.tsv

"""
 This script converts the IKNL discussions data from json to XML and tab-separated values (tsv).
 The XML output will be in the directory threads_dir (one thread per file)
 The dtd for the XML is:

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

import json
import sys
import re
import os

json_file = sys.argv[1]
out_dir = sys.argv[2]
tsvfilename = sys.argv[3]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

threads = dict()  # key is threadid, value is Object thread, to find the thread given the thread id


class Thread:

    def __init__(self,threadid,title,category,ttype):
        self.threadid = threadid
        self.title = title
        self.posts = []
        self.category = category
        self.ttype = ttype

    def addPost(self,post):
        self.posts.append(post)

    def getNrOfPosts(self):
        return len(self.posts)

    def printXML(self,out):
        out.write("<thread id=\""+self.threadid+"\">\n<category>"+self.category+"</category>\n<title>"+self.title+"</title>\n<posts>\n")
        for post in self.posts:
            post.printXML(out)
        out.write("</posts>\n</thread>\n")

    def printTXT(self,out):
        for post in self.posts:
            post.printTXT(out)

    def printTSV(self,out):
        for post in self.posts:
            post.printTSV(self.threadid,out)

def clean_up(text):
    clean_text = re.sub(r"\t","",text)
    clean_text = re.sub("[\r\n]"," ",clean_text)
    clean_text = re.sub("<[^>]+>"," ",clean_text)
    clean_text = re.sub("&nbsp;"," ",clean_text)
    clean_text = re.sub("  +"," ",clean_text)
    #print (clean_text)
    return clean_text

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
        body = clean_up(self.body)
        out.write("<post id=\""+self.postid+"\">\n<author>"+self.author+"</author>\n<timestamp>"+str(self.timestamp)+"</timestamp>\n<parentid>"+self.parentid+"</parentid>\n<body>"+body+"</body>\n<upvotes>"+str(self.ups)+"</upvotes>\n<downvotes>"+str(self.downs)+"</downvotes>\n</post>\n")

    def printTXT(self,out):
        out.write(self.body+"\n")

    def printTSV(self,threadid,out):
        global threads
        body = clean_up(self.body)
        threadforpost = threads[threadid]
        level = ""
        if threadid == self.postid:
            level = "original_post"
        # id	appreciation_count	author	category	created_at	level	thread_id	title	token
        out.write(self.postid+"\t"+str(self.ups)+"\t"+self.author+"\t"+threadforpost.category+"\t"+self.timestamp+"\t"+level+"\t"+threadid+"\t"+threadforpost.title+"\t"+body+"\n")




with open(json_file) as f:
    json_string = ""

    for line in f:
        json_string += line.rstrip()
    parsed_json = json.loads(json_string)
    for item in parsed_json:
        #print (thread)
        threadid = str(item['id'])
        title = item['title']
        category = item['category']
        thread = Thread(threadid,title,category,"")
        threads[threadid] = thread

        author = item['author']
        content = item['body']
        timestamp = item['created_at']
        parentid = ""
        openingpost = Post(threadid,author,timestamp,content,parentid,0,0)

        thread.addPost(openingpost)

        postsarray = item['posts']
        for pitem in postsarray:
            postid = str(pitem['id'])
            author = pitem['author']
            content = pitem['body']
            timestamp = pitem['created_at']
            ups = pitem['appreciation_count']
            parentid = ""
            post = Post(postid,author,timestamp,content,parentid,ups,0)
            thread.addPost(post)


for threadid in threads:
    print(threadid)

    out = open(out_dir+"/"+threadid+".xml","w")
    out.write("<?xml version=\"1.0\"?>\n")
    out.write("<forum type=\"iknl\">\n")
    thread = threads[threadid]

    thread.printXML(out)
    out.write("</forum>\n")
    out.close()


tsv_file = open(tsvfilename,'w')
tsv_file.write("id\tappreciation_count\tauthor\tcategory\tcreated_at\tlevel\tthread_id\ttitle\ttoken\n")

for threadid in threads:
    thread = threads[threadid]
    thread.printTSV(tsv_file)

tsv_file.close()