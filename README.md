# json2xml_iknl.py

This script converts the IKNL discussions data from json to XML and tab-separated values (tsv).
The XML output will be in the directory threads_dir (one thread per file)

```
python json2xml_iknl.py discussions-formatted.json threads_dir KankerNL_posts.tsv
```
# lda.py

The input directory is expected to contain a list of XML files in the unified forum XML format, as created by json2xml_iknl.py (see dtd below).
The LDA model is printed to stdout

```
python lda.py KankerNL_threads
```

# thread_xml2html.py

This script converts a thread XML file to an HTML file that can be viewed in the web browser.
The header and footer HTML files are also included, as well as 2 example input file.

Use:
```
thread_xml2html.py 2ny4u1.xml
thread_xml2html.py 259161.xml
```

The input format is the discussion thread XML format defined for the Discosumo project. The dtd is:

```
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
```

# empowerment_classification.py

Performs supervised classification of labeled forum posts (documentation is in the script). The format of the input file is tab-separated text with in the 7th (final) column the comma-separated labels:
```
threadid	postid	author	timestamp	body	upvotes	labels
```

It makes a split in train and test data and reports Precision, Recall and F-scores per categorie in the test data. It also labels the unlabeled data with empowerment constructs, using all data to train on.

# longitudinal.py

Extracts from the XML formatted forum data all authors with more than 30 posts and prints those posts as a tab separated file, ordered by date (the oldest post first). Format:
```
threadid	postid	author	timestamp	body	upvotes
```

This can be used for longitudinal (qualitative) analysis of specific forum authors.


# License

See the [LICENSE](LICENSE.md) file for license rights and limitations (GNU-GPL v3.0).
