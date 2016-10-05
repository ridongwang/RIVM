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