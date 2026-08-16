"""Scrape text for transcripts for Buzzsprout site and newsletters from
Buttondown site and create ChromaDB collections with text and relevant
metadata.
"""

import argparse
import json
import math
import os
import re
import shutil
import string
import subprocess
import time
import uuid

from datetime import datetime
import bm25s
import chromadb
import requests
import spacy
import Stemmer
from bm25s.tokenization import Tokenizer
from bs4 import BeautifulSoup
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import Page, sync_playwright

from util import strip_punctuation, exact_match_tokenize


def scroll_down(page: Page, scroll_pause: int = 1) -> None:
    """Scroll to bottom of webpage to wait for Javascript to load.

    Args:
        page: playwright Page object
        scroll_pause: number of seconds to wait after scroll (default 1)
    """
    previous_height = None
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(scroll_pause)
    current_height = page.evaluate("document.body.scrollHeight")
    # Check if page height has increased
    while current_height != previous_height:
        previous_height = current_height
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(scroll_pause)
        current_height = page.evaluate("document.body.scrollHeight")


def get_transcript_text(url: str) -> tuple[str, str, str]:
    """Scrape text for one transcript page.

    Args:
        url: link to particular transcript page
    Returns:
        body: text body of page
        title: title of transcript
        timestamp: date of publication
    """
    curl_command = ["curl", "-X", "GET", url]
    try:
        html = subprocess.run(curl_command, capture_output=True, text=True).stdout
    except Exception as e:
        print(f"An error occurred while pulling text for {url}: {e}")
        return None, None
    # We want to exit the function if html wasn't captured even if it
    # didn't throw an error
    if html is None or html == "":
        print(f"HTML output not captured or empty for {url}")
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    # Replacing html tags with spaces rather than just using .text
    # because with .text some words get concatenated when there's a tag
    # but no whitespace which we don't want
    date_str = str(soup.find("time")["datetime"])
    timestamp = int(datetime.strptime(date_str, "%B %d, %Y").timestamp())
    show_notes = re.sub(r"<[^>]+>", " ", str(soup.find(id="show-notes")))
    transcript = re.sub(r"<[^>]+>", " ", str(soup.find(id="transcript")))
    # Replace multiple spaces with single space except new lines
    show_notes = re.sub(r"(?:(?!\n)\s)+", " ", show_notes)
    transcript = re.sub(r"(?:(?!\n)\s)+", " ", transcript)
    body = f"{show_notes} {transcript}"
    title = soup.title.text
    return body, title, timestamp


def get_newsletter_text(url: str) -> tuple[str, str, str]:
    """Scrape text for one newsletter page.

    Args:
        url: link to particular newsletter page
    Returns:
        body: text body of page
        title: title of newsletter
        timestamp: date of publication
    """
    try:
        html = requests.get(url).content
    except Exception as e:
        print(f"An error occurred while pulling text for {url}: {e}")
        return None, None
    # We want to exit the function if html wasn't captured even if it
    # didn't throw an error
    if html is None or html == "":
        print(f"HTML output not captured or empty for {url}")
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    date_str = re.sub(r"<[^>]+>", " ", str(soup.find("date"))).strip()
    timestamp = int(datetime.strptime(date_str, "%B %d, %Y").timestamp())
    # Replacing html tags with spaces rather than just using .text
    # because with .text some words get concatenated when there's a tag
    # but no whitespace which we don't want
    body = re.sub(r"<[^>]+>", " ", str(soup.find("div", class_="email-body-content")))
    # Replace multiple spaces with single space except new lines
    body = re.sub(r"(?:(?!\n)\s)+", " ", body)
    # Replace extra bluesky embedding text with space
    bsky_pattern = r".bluesky-embed {(.*?){ display: none !important; }"
    body = re.sub(bsky_pattern, " ", body)
    title = soup.find("h1", class_="subject").text
    return body, title, timestamp


def get_transcript_links(main_url: str) -> list[str]:
    """Scrape all links for transcripts.

    Args:
        main_url: link for transcript homepage
    Returns:
        link: list of individual transcript links
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(main_url)
        scroll_down(page)
        html = page.inner_html("*")
        browser.close()
    soup = BeautifulSoup(html, "html.parser")
    links = [
        a.get("href")
        for a in soup.find_all("a")
        if a.has_attr("class") and a["class"][0] == "w-full"
    ]
    links = [f"https://www.buzzsprout.com/{link}" for link in links]
    links.reverse()
    return links


def get_newsletter_links(main_url: str) -> list[str]:
    """Scrape all links for newsletters.

    Args:
        main_url: link for newsletter homepage
    Returns:
        link: list of individual newsletter links
    """
    links = []
    page_number = 1
    increment = True
    while increment == True:
        main_url_plus_page = f"{main_url}?page={page_number}"
        html = requests.get(main_url_plus_page).content
        soup = BeautifulSoup(html, "html.parser")
        new_links = [
            item.get("href") 
            for item in soup.find_all("a")
            if item.has_attr("class") and item["class"][0] == "email-link"
        ]
        if new_links[0] in links:
            increment = False
        else:
            links += new_links
            page_number += 1
    links.reverse()
    return links


def get_batch_indices(
        docs: list[str],
        max_batch_size: int = 5461,
    ) -> list[list[int, int]]:
    """Split documents/chunks into batches.

    Max batch size for adding to chromadb collection is 5461, so split into
    more batches if needed.

    Args:
        docs: list of text from documents/chunks
        max_batch_size: max amount of docs we can add at a time (default 5461)
    Returns:
        batch_indices: list of indices for where each batch begins/ends
    """
    num_docs = len(docs)
    num_batches = math.ceil(num_docs / max_batch_size)
    batch_indices = [
        [max_batch_size * i, max_batch_size * (i + 1)]
        for i in range(num_batches)
    ]
    batch_indices[-1][1] = num_docs
    return batch_indices


def get_doc_piece_map(body: str) -> dict[str, list[str]]:
    """Create mapping of 'doc piece' to not stemmed tokens from the 'doc piece'.

    'Doc pieces' are portions of document split by whitespace. We need to have
    the original (not stemmed) versions of the tokens of each doc piece in order
    to do keyword highlighting, because there are instances where a doc piece
    contains muliple tokens. Needs to not be stemmed to match original text.

    Args:
        body: text of document
    Returns:
        doc_piece_map: dictionary mapping text of doc piece to not stemmed
                       tokens of doc piece
    """
    doc_pieces = set(body.split())
    doc_piece_map = {
        doc_piece: list(
            bm25s.tokenize(doc_piece, stopwords=None, stemmer=None)[1]
        )
        for doc_piece in doc_pieces
    }
    return doc_piece_map


def get_stem_token_map(body: str, method: str) -> dict[str, list[str]]:
    """Create mapping of stemmed tokens to not stemmed tokens.
    
    Need this for keyword highlighting because we need to find which portions
    of document match stemmed query tokens but we need to highlight the not
    stemmed counterpart.

    Args:
        body: text of document
    Returns:
        stem_token_map: dictionary mapping each stemmed token from document to
                        its not stemmed counterparts
    """
    stem_token_map = {}
    if method == "stem":
        bm25_tokens_output = bm25s.tokenize(
            body.strip(), stopwords=None, stemmer=None
        )
        bm25_tokens = list(bm25_tokens_output[1])
        for token in bm25_tokens:
            stem = stemmer.stemWord(token)
            if stem in stem_token_map:
                stem_token_map[stem].append(token)
            else:
                stem_token_map[stem] = [token]
    return stem_token_map


def get_spacy_lists(body: str) -> tuple[list[list[str]], list[list[str]]]:
    """Create lists of spacy tokens and lemmas needed later for keyword
    matching.

    Args:
        body: text of document
    Returns:
        tokens_full: list of lists of spaCy tokens corresponding to each doc_piece
        lemmas_full: list of lists of spaCy lemmas corresponding to each doc_piece
    """
    tokens_full = []
    lemmas_full = []
    doc_pieces = body.split()
    # Replace any whitespace with single space
    body = " ".join(body.split())
    doc = nlp(body.strip().lower())
    token_list = [token for token in doc]
    lemma_list = [token.lemma_ for token in doc]
    for doc_piece in doc_pieces:
        temp = doc_piece.lower()
        loop_through = True
        add_to_tokens = []
        add_to_lemmas = []
        while loop_through == True:
            if len(token_list)==0:
                loop_through = False
            elif token_list[0].text in temp:
                token_to_add = token_list.pop(0)
                add_to_tokens.append(token_to_add.text)
                lemma_to_add = lemma_list.pop(0)
                add_to_lemmas.append(lemma_to_add)
            else:
                loop_through = False
            if loop_through == False:
                tokens_full.append(add_to_tokens)
                lemmas_full.append(add_to_lemmas)
    return tokens_full, lemmas_full


def add_to_collection(
        collection: Collection,
        links: list[str],
        category: str,
        mode: str = "rewrite",
        chunking: bool = False,
    ) -> None:
    """Add documents and their metadata to chromadb collection.

    Args:
        collection: chromadb collection
        links: list of transcript or newsletter links to add
        category: "transcripts" or "newsletters"
        mode: "rewrite" rewrites entire collection, "update" checks for links
              that aren't already in collection and then adds only those
        chunking: True if splitting document into chunks (only required for
                  dense vectors)
    Returns:
        updated_bool: True if an update was made, False if not
    """
    docs_all = []
    ids = []
    metadata = []
    if mode == "update":
        # Narrow down links to only the new ones
        existing_links = set(
            item["link"]
            for item in collection.get(where={"category": category})["metadatas"]
        )
        links = [link for link in links if link not in existing_links]
        if len(links) == 0:
            print(f"No new {category} to add")
            updated_bool = False
            return updated_bool
    for link in links:
        print(link)
        if category == "transcripts":
            body, title, timestamp = get_transcript_text(link)
            if not (body and title):
                print(f"Did not add {link} to collection")
                continue
        elif category == "newsletters":
            body, title, timestamp = get_newsletter_text(link)
            if not (body and title):
                print(f"Did not add {link} to collection")
                continue
        if chunking:
            chunks = chunk_splitter.split_text(body)
            for chunk in chunks:
                docs_all.append(chunk)
                ids.append(f"{link}-{uuid.uuid5(uuid.NAMESPACE_DNS, chunk)}")
                doc_piece_map = get_doc_piece_map(chunk)
                stem_token_map = get_stem_token_map(chunk, "stem")
                spacy_tokens, spacy_lemmas = get_spacy_lists(chunk)
                metadata.append(
                    {
                        "title": title,
                        "link": link,
                        "timestamp": timestamp,
                        "category": category,
                        "doc_piece_map": json.dumps(doc_piece_map),
                        "stem_token_map": json.dumps(stem_token_map),
                        "spacy_tokens": json.dumps(spacy_tokens),
                        "spacy_lemmas": json.dumps(spacy_lemmas),
                    }
                )
        else:
            docs_all.append(body)
            ids.append(f"{link}-{uuid.uuid5(uuid.NAMESPACE_DNS, body)}")
            doc_piece_map = get_doc_piece_map(body)
            stem_token_map = get_stem_token_map(body, "stem")
            spacy_tokens, spacy_lemmas = get_spacy_lists(body)
            metadata.append(
                {
                    "title": title,
                    "link": link,
                    "timestamp": timestamp,
                    "category": category,
                    "doc_piece_map": json.dumps(doc_piece_map),
                    "stem_token_map": json.dumps(stem_token_map),
                    "spacy_tokens": json.dumps(spacy_tokens),
                    "spacy_lemmas": json.dumps(spacy_lemmas),
                }
            )
    batch_indices = get_batch_indices(docs_all)
    for indices in batch_indices:
        collection.upsert(
            documents=docs_all[indices[0]:indices[1]],
            metadatas=metadata[indices[0]:indices[1]],
            ids=ids[indices[0]:indices[1]],
        )
    updated_bool = True
    return updated_bool


def save_bm25_index(corpus: list[str], keyword_match_method: str, category: str):
    """Save BM25 indexes ahead of time so app is faster, especially helpful
    for lemma keyword match option.

    Args:
        corpus: list of text of documents
        keyword_match_method: "Exact", "Lemma", or "Stem"
        category: "transcripts" or "newsletters"
    """
    dir_name = f"bm25_indexes/bm25_index_{keyword_match_method.lower()}_{category}"
    # Default parameter values are k1=1.5, b=0.75
    retriever = bm25s.BM25(corpus=corpus)
    if keyword_match_method == "Exact":
        splitter = lambda doc: exact_match_tokenize(doc)
        tokenizer = Tokenizer(
            stemmer=None, stopwords=None, splitter=splitter
        )
        retriever.index(tokenizer.tokenize(corpus, return_as="tuple"))
    elif keyword_match_method == "Lemma":
        splitter = lambda input_text: [token for token in nlp(input_text.lower())]
        keyword_match_function = lambda token: str(token.lemma_)
        tokenizer = Tokenizer(
              stemmer=keyword_match_function, stopwords=None, splitter=splitter
        )
        retriever.index(tokenizer.tokenize(corpus, return_as="tuple"))
    elif keyword_match_method == "Stem":
        retriever.index(bm25s.tokenize(corpus, stemmer=Stemmer.Stemmer("english")))
    retriever.save(dir_name, corpus=corpus)


if __name__ == "__main__":
    start = time.time()
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    stemmer = Stemmer.Stemmer("english")
    persist_dir = "chroma_db"
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.config.Settings(allow_reset=True),
    )
    nlp = spacy.load("en_core_web_sm")

    # Command line argument for mode
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True,
                        help="update or rewrite")
    args = parser.parse_args()
    # "update" only adds documents with links that don't already exist in the
    # database, "rewrite" deletes entire database before creating a new one
    mode = args.mode
    if mode == "rewrite":
        client.reset()

    # Create/get collections
    collection_full = client.get_or_create_collection(
        name="maiht3k_full",
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection_chunks = client.get_or_create_collection(
        name="maiht3k_chunks",
        configuration={"hnsw": {"space": "cosine"}},
    )
    
    # Transcripts
    transcript_url = "https://www.buzzsprout.com/2126417/episodes/"
    transcript_links = get_transcript_links(transcript_url)
    transcripts_updated_bool = add_to_collection(
        collection_full, transcript_links, "transcripts", mode
    )
    if transcripts_updated_bool:
        transcripts_updated_bool = add_to_collection(
            collection_chunks, transcript_links, "transcripts", mode, chunking=True
        )
    
    # Newsletters
    newsletter_url = "https://buttondown.com/maiht3k/archive/"
    newsletter_links = get_newsletter_links(newsletter_url)
    newsletters_updated_bool = add_to_collection(
        collection_full, newsletter_links, "newsletters", mode
    )
    if newsletters_updated_bool:
        newsletters_updated_bool = add_to_collection(
            collection_chunks, newsletter_links, "newsletters", mode, chunking=True
        )
    
    # Redo BM25 indexes for new full corpus and save ahead of time
    if transcripts_updated_bool or newsletters_updated_bool:
        # Delete existing directories and their contents first
        match_methods = ["Exact", "Lemma", "Stem"]
        categories = ["all", "transcripts", "newsletters"]
        paths = [
            f"bm25_indexes/bm25_index_{match_method.lower()}_{category}"
            for match_method in match_methods
            for category in categories
        ]
        for path in paths:
            if os.path.isdir(path):
                shutil.rmtree(path)
        # Save BM25 indexes for each match method and category separately
        for category in categories:
            if category == "all":
                corpus = collection_full.get()["documents"]
            else:
                corpus = collection_full.get(where={"category": category})["documents"]
            for match_method in match_methods:
                save_bm25_index(corpus, match_method, category)

    print(collection_full.count())
    print(collection_chunks.count())
    print(time.time() - start)

