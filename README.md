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

Performs supervised classification of labeled forum posts. The format of the input file is tab-separated text with in the 7th (final) column the comma-separated labels:
```
threadid	postid	author	timestamp	body	upvotes	labels
```

It makes a 50-50 split in train and test data and reports Precision, Recall and F-scores per categorie in the data (for categories with at least 10 examples in the test set).
It also prints the list of 10 most important features (words or character 4-grams) per category according to the LogisticRegression model.

# License

See the [LICENSE](LICENSE.md) file for license rights and limitations (GNU-GPL v3.0).
