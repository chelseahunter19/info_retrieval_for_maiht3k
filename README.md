# Information Retrieval for MAIHT3k

This repo contains code for the information retrieval system for searching MAIHT3k podcast transcripts and newsletters. There are two main components: a database update script and a streamlit app search interface.

## Database update script

scrape.py pulls the text of the transcripts and newsletters from the Buzzsprout and Buttondown websites and saves them in a ChromaDB database. It also does additional pre-processing needed for the Streamlit app. The script can be run in either “rewrite” or “update” mode as follows:

`python scrape.py -m rewrite`

will delete the entire database before recreating it.

`python scrape.py -m update`

will check for new transcripts and newsletters and add them to the existing database.

This script also creates BM25 indexes that are necessary for the BM25 search method in the Streamlit app, and it saves them as separate files.

## Streamlit app search interface

The streamlit app can be run using the command:

`streamlit run app.py`

