# coding=utf-8
# thread_xml2html.py 2ny4u1.xml
# thread_xml2html.py 259161.xml

import sys
import re
import xml.etree.ElementTree as ET

hierarchy = False


xmlfile = sys.argv[1]
htmlfile = xmlfile.replace("xml","html")
out = open(htmlfile,'w')

with open("header.html",'r') as header:
    for line in header:
        if re.match(".*<title>.*",line):
            out.write("<title>"+xmlfile+"</title>")
        else:
            out.write(line)



children = dict() # key is parentid, value is list of children ids
postids = dict() # key is postid, value is post

tree = ET.parse(xmlfile)
root = tree.getroot()
id_of_firstpost = ""
forumtype = root.get('type')
print (forumtype)
if forumtype == "viva":
    hierarchy = False
if forumtype == "reddit":
    hierarchy = True
#print (forumtype, hierarchy)


out.write('<div class="col-sm-6">\n')
out.write('<div class="list-group">\n')

list_of_posts = list()
maxnrofposts = 50
for thread in root:
    for posts in thread.findall('posts'):
        firstpost = posts.findall('post')[0]
        #id_of_firstpost = firstpost.get('id')
        postcount = 0
        for post in posts.findall('post'):
            postcount += 1
            if postcount > 50:
                break
            list_of_posts.append(post)
            postid = post.get('id')
            postids[postid] = post
            parentid = post.find('parentid').text
            if parentid is not None:
                children_of_parent = list()
                if parentid in children:
                    children_of_parent = children[parentid]
                children_of_parent.append(postid)
                #print ("parent:",parentid,"child:",postid)
                children[parentid] = children_of_parent

def replace_quote(postcontent):
    # for Viva forum data
    adapted = ""

    postcontent = re.sub("\n"," ",postcontent)
    #print (postcontent)
    blocks = re.split("<br>",postcontent)
    #print (blocks)
    # first, find the block with the quote:
    bi=0
    bc = len(blocks)
    quoteblocki = 4
    while bi < bc:
        if " schreef op " in blocks[bi]:
            #print (blocks[bi])
            quoteblocki = bi
            break

        # print until the quote:
        if not re.match('^>',blocks[bi]):
            adapted += blocks[bi]+"<br>\n"

        bi += 1
    blocks[quoteblocki] = re.sub("^>","",blocks[quoteblocki])
    blocks[quoteblocki] = re.sub("\(http://.*\):","",blocks[quoteblocki])
    quote = blocks[quoteblocki]+"<br>\n"
    if len(blocks) > quoteblocki+1:
        quote += blocks[quoteblocki+1]+"<br>\n"
    adapted += "<div style='background-color:rgb(240,240,240);padding-left:4em;padding-right:4em'>"+quote+"</div><br>\n"

    bi = quoteblocki+2
    while bi < bc:
        adapted += blocks[bi]+"<br>\n"
        bi += 1


    return adapted

row=0

postidperrow = dict()
openingpostwithauthor = ""
def print_post(currentpostid,indent):
    global row
    global openingpostwithauthor
    currentpost = postids[currentpostid]
    author = currentpost.find('author').text
    timestamp = currentpost.find('timestamp').text
    bodyofpost = currentpost.find('body').text

    if bodyofpost is None:
        bodyofpost = ""
    if re.match(".*http://[^ ]+\n[^ ]+.*",bodyofpost):
        bodyofpost = re.sub("(http://[^ ]+)\n([^ ]+)",r"\1\2",bodyofpost)

    bodyofpost = re.sub("\"","&#34;",bodyofpost)
    #bodyofpost = re.sub("\'","&#39;",bodyofpost)
    #bodyofpost = re.sub("\'","\\\'",bodyofpost)
    bodyofpost = re.sub("\n *\n","<br>\n",bodyofpost)
    #print (currentpostid, bodyofpost)
    if " schreef op " in bodyofpost:
        bodyofpost = replace_quote(bodyofpost)

    bodyofpost = re.sub("\n"," ",bodyofpost)

    if "smileys" in bodyofpost:
        bodyofpost = re.sub(r'\((http://forum.viva.nl/global/(www/)?smileys/.*.gif)\)','',bodyofpost)


    upvotefield = currentpost.find('upvotes')
    downvotefield = currentpost.find('downvotes')

    bodywithauthor = '<b>'+author+'</b> @ '+timestamp+':<br> '+bodyofpost

    if currentpostid == "0":
        openingpostwithauthor = bodywithauthor
    out.write("<tr>")
    row += 1
    out.write('<td>')
    if row==1:
        bgcolor="rgb(240,240,240)"
    else:
        bgcolor="white"
    out.write('<a href="#" class="list-group-item" style="background-color:'+bgcolor+'; padding-left:'+str(indent)+'em" ')
    out.write('>')
    out.write(bodywithauthor)
    out.write('\n')
    postidperrow[row] = currentpostid
    if upvotefield is not None and downvotefield is not None:

        upvotes = currentpost.find('upvotes').text
        downvotes = currentpost.find('downvotes').text
        score = int(upvotes)-int(downvotes)
        out.write('<div style="font-size:8pt;border-style:none">'+str(score)+' upvotes</div>\n')

    out.write("</a></td></tr>\n")



def print_children(leaf,indent): # leaf is a post id, indent is the size of the indentation, row is the nth row of the list
    global row
    indent += 3
    #print indent,leaf
    currentpostid = leaf
    print_post(currentpostid,indent)

    if leaf in children:
        children_of_leaf = children[leaf]
        #print "has children", children_of_leaf
        for child in children_of_leaf:
            print_children(child,indent)

#print_children(id_of_firstpost,"")

title=""
noofposts = 0
for thread in root:
    threadid = thread.get('id')
    category = thread.find('category').text
    title = thread.find('title').text
    out.write('<a href="#" class="list-group-item active" style="padding-left:2em">[category: '+category+']<br><br>\n<b>'+title+'</b></a>\n')
    out.write('<table border=1 width="100%">\n')
    #for posts in thread.findall('posts'):
        #id_of_firstpost = firstpost.get('id')
    id_of_firstpost = list_of_posts[0].get('id')
    #noofposts = len(list_of_posts)
    if hierarchy:
        print_children(id_of_firstpost,0)
    else:
        for post in list_of_posts:
            postid = post.get('id')
            print_post(postid,2)

    out.write("</table>\n")

    out.write('</div>\n</div>\n')


    with open("footer.html",'r') as header:
        for line in header:
            out.write(line)

    sys.stderr.write("Output written to "+out.name+"\n")

    out.close()