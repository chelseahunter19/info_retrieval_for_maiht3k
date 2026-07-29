"""Streamlit app for searching transcripts and newsletters pulling from
ChromaDB collections.
"""

import json
import string
import time
from itertools import chain
from operator import itemgetter

import bm25s
import chromadb
import spacy
import Stemmer
import streamlit as st
from bm25s.tokenization import Tokenized, Tokenizer
from chromadb.api.models.Collection import Collection


def run_boolean(
        query_mod: list,
        collection: Collection,
        match_method,
        order: str = "chronological",
    ) -> list[str]:
    """Run boolean search for chronological sort methods.

    Args:
        query_mod: tokens from query modified according to match method
        collection: chromadb collection
        match_method: method for keyword matching--exact, lemma, or stem
        order: chronological or reverse chronological
    Returns:
        titles: list of titles that contain query tokens in order
    """
    collection_filtered = collection.get(where=category_filter)
    ids = collection_filtered["ids"]
    title_timestamp_map = {}
    for target_id in ids:
        temp = collection.get(ids=target_id)
        doc = temp["documents"][0]
        title = temp["metadatas"][0]["title"]
        if match_method == "Exact":
            doc_pieces = doc.split()
            tokens = [
                doc_piece.strip(string.punctuation).lower() for doc_piece in doc_pieces
            ]
        elif match_method == "Lemma":
            lemma_lists = json.loads(temp["metadatas"][0]["lemmas"])
            tokens = [item for sublist in lemma_lists for item in sublist]
        elif match_method == "Stem":
            tokens = set(bm25s.tokenize(doc, stemmer=keyword_match_function)[1].keys())
        if any(item in tokens for item in query_mod):
            title_timestamp_map[title] = temp["metadatas"][0]["timestamp"]
    if order == "chronological":
        title_timestamp_map = dict(
            sorted(title_timestamp_map.items(), key=itemgetter(1))
        )
    elif order == "reverse_chronological":
        title_timestamp_map = dict(
            sorted(title_timestamp_map.items(), key=itemgetter(1), reverse=True)
        )
    titles = list(title_timestamp_map.keys())
    return titles


def run_query_sparse_vectors(
        query: str,
        collection: Collection,
        match_method: str,
    ) -> dict[str, float]:
    """Run query and rank document relevance using sparse vectors.

    Args:
        query: user input query
        collection: chromadb collection
        match_method: method for keyword matching--exact, lemma, or stem
    Returns:
        doc_score_map: mapping from title of document to score
    """
    collection_filtered = collection.get(where=category_filter)
    corpus = collection_filtered["documents"]
    num_docs = len(corpus)
    titles = [
        item["title"]
        for item in collection_filtered["metadatas"]
    ]
    doc_title_map = dict(zip(corpus, titles))

    if match_method == "Exact":
        # Default parameter values are k1=1.5, b=0.75
        retriever = bm25s.BM25(corpus=corpus)
        splitter = lambda doc: [
            doc_piece.strip(string.punctuation).lower() for doc_piece in doc.split()
        ]
        tokenizer = Tokenizer(
            stemmer=keyword_match_function, stopwords=None, splitter=splitter
        )
        query_tokenizer = Tokenizer(
            stemmer=keyword_match_function, stopwords=None, splitter=splitter
        )
        retriever.index(
            tokenizer.tokenize(corpus, return_as="tuple", update_vocab=True)
        )
        query_tokenized = query_tokenizer.tokenize(
            [query], return_as="tuple", update_vocab=True
        )
    elif match_method == "Lemma":
        splitter_query = lambda input_text: [token for token in nlp(input_text)]
        query_tokenizer = Tokenizer(
            stemmer=keyword_match_function, stopwords=None, splitter=splitter_query
        )
        query_tokenized = query_tokenizer.tokenize(
            [query], return_as="tuple", update_vocab=True
        )
        retriever = bm25s.BM25.load("bm25_lemmas", load_corpus=True)
    elif match_method == "Stem":
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(bm25s.tokenize(corpus, stemmer=keyword_match_function))
        query_tokenized = bm25s.tokenize(query, stemmer=keyword_match_function)

    docs_ordered, scores = retriever.retrieve(
        query_tokenized, k=num_docs
    )
    if match_method == "Lemma":
        titles_ordered = [doc_title_map[doc["text"]] for doc in docs_ordered[0]]
    else:
        titles_ordered = [doc_title_map[doc] for doc in docs_ordered[0]]
    doc_score_map = dict(zip(titles_ordered, scores[0]))
    return doc_score_map


def run_query_dense_vectors(
        query: str,
        collection: Collection,
    ) -> dict[str, float]:
    """Run query and rank document relevance using dense vectors.

    Args:
        query: text of user query
        collection: chromadb collection
    Returns:
        doc_score_map: mapping from title of document to score
    """
    # ChromaDB collections use the all-MiniLM-L6-v2 embedding model by default
    results = collection.query(
        query_texts=[query],
        where=category_filter,
        n_results=len(collection.get(where=category_filter)["ids"]),
    )
    id_score_map = dict(zip(results["ids"][0], results["distances"][0]))
    # Convert from cosine distance to cosine similarity
    # cosine similarity = 1 - cosine distance
    id_score_map = {key: 1 - id_score_map[key] for key in id_score_map}
    doc_score_map = aggregate_scores_at_doc_level(
        collection, id_score_map
    )
    doc_score_map = dict(
        sorted(doc_score_map.items(), key=itemgetter(1), reverse=True)
    )
    return doc_score_map


def run_hybrid(
        titles_sparse: list[str],
        titles_dense: list[str],
    ) -> dict[str, float]:
    """Run hybrid ranking.

    Uses Reciprocal Rank Fusion to combine sparse and dense rankings to get
    hybrid score.

    Args:
        titles_sparse: list of titles in order of sparse ranking
        titles_dense: list of titles in order of dense ranking
    Returns:
        hybrid_scores: mapping of titles to hybrid score, sorted in order of
                       best to worst
    """
    # rank dictionary: map of title to rank number
    rank_sparse = dict(zip(titles_sparse, range(len(titles_sparse))))
    rank_dense = dict(zip(titles_dense, range(len(titles_dense))))
    hybrid_scores = {}
    k = 60
    w_sparse = 1
    w_dense = 1
    for key in rank_sparse:
        hybrid_scores[key] = (
            (w_sparse / (k + rank_sparse[key]))
            + (w_dense / (k + rank_dense[key]))
        )
    hybrid_scores = dict(
        sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    )
    return hybrid_scores


def aggregate_scores_at_doc_level(
        collection: Collection,
        id_score_map: dict[str, float],
    ) -> dict[str, float]:
    """Aggregate chunk scores for each document so we have document-level scores.
    
    Args:
        collection: chromadb collection
        id_score_map: mapping from ids to scores from running dense
    Returns:
        doc_score_map: mapping from document title to aggregated score
    """
    metadata = collection.get(where=category_filter)["metadatas"]
    doc_titles = list(set(chunk["title"] for chunk in metadata))
    doc_score_map = {}
    for doc_title in doc_titles:
        # We won't need category_filter here because we've already narrowed down
        # the documents
        doc_chunk_ids = collection.get(where={"title": doc_title})["ids"]
        scores = [id_score_map[doc_chunk_id] for doc_chunk_id in doc_chunk_ids]
        #doc_score_map[doc_title] = sum(scores)/len(scores)
        doc_score_map[doc_title] = max(scores)
    return doc_score_map


def write_keyword_highlighting_exact_match(
        collection: Collection,
        target_id: str,
        query_originals: list[str],
        query_mod: list[str],
        match_method: str,
    ) -> None:
    """Write snippets of document with keywords highlighted.

    Args:
        collection: chromadb collection
        target_id: id of document/chunk that we're displaying keywords for
        query_originals
        query_mod
        match_method: exact, lemma, or stem
    """
    id_collection = collection.get(ids=[target_id])
    doc = id_collection["documents"][0].strip()
    doc_pieces = doc.split()
    positions = [
        i
        for i in range(len(doc_pieces))
        if doc_pieces[i].strip(string.punctuation).lower() in query_mod
    ]
    # Highlight the original token in each of those positions
    for position in positions:
        if doc_pieces[position].strip(string.punctuation).lower() in query_mod:
            token = doc_pieces[position].strip(string.punctuation)
            doc_pieces[position] = doc_pieces[position].replace(
                token, f"<mark><strong>{token}</mark></strong>"
            )
    # Group highlighted portions into snippets to display
    span_size = 10
    span_positions = [[i - span_size, i + span_size] for i in positions]
    positions_grouped = []
    for i in range(len(span_positions)):
        if i == 0:
            positions_grouped.append(span_positions[i])
        else:
            # If span overlap (last position of previous span greater than
            # or equal to first position of current span)
            if span_positions[i - 1][1] >= span_positions[i][0]:
                positions_grouped[-1][1] = span_positions[i][1]
            else:
                positions_grouped.append(span_positions[i])
    for position_group in positions_grouped:
        start = max(0, position_group[0])
        end = min(position_group[-1], len(doc.split(" ")) - 1)
        span_items = doc_pieces[start:end + 1]
        span = " ".join(span_items)
        st.html(span)


def write_keyword_highlighting_lemma_match(
        collection: Collection,
        target_id: str,
        query_originals: list[str],
        query_mod: list[str],
        match_method: str,
    ) -> None:
    """Write snippets of document with keywords highlighted.

    Args:
        collection: chromadb collection
        target_id: id of document/chunk that we're displaying keywords for
        query_originals
        query_mod
        match_method: exact, lemma, or stem
    """
    id_collection = collection.get(ids=[target_id])
    doc = id_collection["documents"][0].strip()
    doc_pieces = doc.split()
    originals = json.loads(
        id_collection["metadatas"][0]["originals_spacy"]
    )
    modified = json.loads(
        id_collection["metadatas"][0]["lemmas"]
    )
    positions = [
        i
        for i in range(len(doc_pieces))
        if any(
            mod.lower() in query_mod for mod in modified[i]
        )
    ]
    # Highlight the original token in each of those positions
    for position in positions:
        for i in range(len(modified[position])):
            if modified[position][i].lower() in query_mod:
                token = originals[position][i]
                doc_pieces[position] = doc_pieces[position].replace(
                    token, f"<mark><strong>{token}</mark></strong>"
                )
    # Group highlighted portions into snippets to display
    span_size = 10
    span_positions = [[i - span_size, i + span_size] for i in positions]
    positions_grouped = []
    for i in range(len(span_positions)):
        if i == 0:
            positions_grouped.append(span_positions[i])
        else:
            # If span overlap (last position of previous span greater than
            # or equal to first position of current span)
            if span_positions[i - 1][1] >= span_positions[i][0]:
                positions_grouped[-1][1] = span_positions[i][1]
            else:
                positions_grouped.append(span_positions[i])
    for position_group in positions_grouped:
        start = max(0, position_group[0])
        end = min(position_group[-1], len(doc.split(" ")) - 1)
        span_items = doc_pieces[start:end + 1]
        span = " ".join(span_items)
        st.html(span)


def write_keyword_highlighting_stem_match(
        collection: Collection,
        target_id: str,
        query_originals: list[str],
        query_mod: list[str],
        match_method: str,
    ) -> None:
    """Write snippets of document with keywords highlighted.

    Args:
        collection: chromadb collection
        target_id: id of document/chunk that we're displaying keywords for
        query_originals
        query_mod
        match_method: exact, lemma, or stem
    """
    id_collection = collection.get(ids=[target_id])
    doc = id_collection["documents"][0].strip()
    doc_pieces = doc.split()
    # Mapping of doc piece to not stemmed tokens
    doc_piece_map = json.loads(
        id_collection["metadatas"][0]["doc_piece_map"]
    )
    # Mapping of stemmed tokens to not stemmed tokens
    token_map = json.loads(
        id_collection["metadatas"][0]["stem_token_map"]
    )
    # Keep only query tokens that are in document
    query_tokens = query_mod
    query_tokens = [
        token for token in query_tokens if token in token_map.keys()
    ]
    # Not stemmed counterparts of the query tokens
    not_stemmed_options = set(
        chain.from_iterable([token_map[stemmed] for stemmed in query_tokens])
    )
    # Sort longest to shortest so we highlight longest first
    # (e.g. cats before cat)
    not_stemmed_options = sorted(
        list(not_stemmed_options), key=len, reverse=True
    )
    # Find the positions of doc pieces that contain one of the
    # not stemmed tokens
    positions = [
        i
        for i, item in enumerate(doc_pieces)
        if any(
            not_stemmed in doc_piece_map[item] for not_stemmed in not_stemmed_options
        )
    ]
    # Highlight the not stemmed token in each of those positions
    for position in positions:
        for not_stemmed in not_stemmed_options:
            doc_pieces[position] = doc_pieces[position].replace(
                not_stemmed, f"<mark><strong>{not_stemmed}</mark></strong>"
            )
    # Group highlighted portions into snippets to display
    span_size = 10
    span_positions = [[i - span_size, i + span_size] for i in positions]
    positions_grouped = []
    for i in range(len(span_positions)):
        if i == 0:
            positions_grouped.append(span_positions[i])
        else:
            # If span overlap (last position of previous span greater than
            # or equal to first position of current span)
            if span_positions[i - 1][1] >= span_positions[i][0]:
                positions_grouped[-1][1] = span_positions[i][1]
            else:
                positions_grouped.append(span_positions[i])
    for position_group in positions_grouped:
        start = max(0, position_group[0])
        end = min(position_group[-1], len(doc.split(" ")) - 1)
        span_items = doc_pieces[start:end + 1]
        span = " ".join(span_items)
        st.html(span)


def print_results(
        collection: Collection,
        top_docs: list[str],
        query_originals: list[str],
        query_mod: list[str],
        match_method: str,
    ) -> None:
    """Print top search results.

    Includes title, link, score, and text with keywords from query highlighted.

    Args:
        collection: chromadb collection
        top_docs: list of titles of top ranking documents
        doc_score_map: mapping of document title to score
        query_orignals
        query_mod
    """
    for top_doc in top_docs:
        # We won't need category_filter here because we've already narrowed down
        # the documents
        collection_subset = collection.get(where={"title": top_doc})
        metadata_temp = collection_subset["metadatas"][0]
        # Write results
        st.markdown(f"##### [{metadata_temp['title']}]({metadata_temp['link']})")
        with st.expander("Keyword matches"):
            ids = collection_subset["ids"]
            for target_id in ids:
                if match_method == "Exact":
                    write_keyword_highlighting_exact_match(
                        collection, target_id, query_originals, query_mod, match_method
                    )
                elif match_method == "Lemma":
                    write_keyword_highlighting_lemma_match(
                        collection, target_id, query_originals, query_mod, match_method
                    )
                elif match_method == "Stem":
                    write_keyword_highlighting_stem_match(
                        collection, target_id, query_originals, query_mod, match_method
                    )


if __name__ == "__main__":
    start = time.time()
    stemmer = Stemmer.Stemmer("english")
    client = chromadb.PersistentClient(path="chroma_db")
    collection_full = client.get_collection(name="maiht3k_full")
    collection_chunks = client.get_collection(name="maiht3k_chunks")
    category_filter = None
    # Decrease vertical white
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.5rem;    /* Default is ~6rem */
            padding-bottom: 2.5rem;
            padding-left: 0rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    # Decrease vertical white space between elements
    st.markdown(
        """
        <style>
        [data-testid="stVerticalBlock"] {
            gap: 0.7rem; /* Default is usually 1rem or higher */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    nlp = spacy.load("en_core_web_sm")
    st.title("Information retrieval for MAIHT3k")
    
    # Define options
    with st.sidebar:
        st.sidebar.header("Options")
        n = int(st.text_input("Number of results", "5"))
        cat_options = ["Podcast transcripts", "Newsletters"]
        # Category options
        selected_cat_options = st.multiselect(
            "Select one or more categories:",
            cat_options,
            default=cat_options,
        )
        if set(selected_cat_options) == set(cat_options):
            category_filter = {
                "$or": [{"category": "transcripts"}, {"category": "newsletters"}]
            }
        elif selected_cat_options == ["Podcast transcripts"]:
            category_filter = {"category": "transcripts"}
        elif selected_cat_options == ["Newsletters"]:
            category_filter = {"category": "newsletters"}
        # Keyword match options
        selected_keyword_match_method = st.selectbox(
            "Select keyword match method:", ["Exact", "Lemma", "Stem"]
        )
        if selected_keyword_match_method == "Exact":
            keyword_match_function = None
        elif selected_keyword_match_method == "Lemma":
            keyword_match_function = lambda token: str(token.lemma_).lower()
        elif selected_keyword_match_method == "Stem":
            keyword_match_function = Stemmer.Stemmer("english")
        # Ordering options
        search_methods = [
            "Oldest to newest",
            "Newest to oldest",
            "Sparse vectors (BM25)",
            "Dense vectors (embedding model)",
            "Hybrid",
        ]
        selected_search_method = st.selectbox(
            "Select ordering method:", search_methods
        )

    # Run query
    query = st.text_input("Enter search query", "")
    if query and category_filter:
        if selected_keyword_match_method == "Exact":
            query_originals = [word.strip(string.punctuation) for word in query.split()]
            query_mod = [word.lower() for word in query_originals]
        elif selected_keyword_match_method == "Lemma":
            doc_query = nlp(query.strip())
            query_originals = [str(token) for token in doc_query]
            query_mod = [str(token.lemma_).lower() for token in doc_query]
        elif selected_keyword_match_method == "Stem":
            query_originals = list(
                bm25s.tokenize(query, stopwords=None, stemmer=None)[1].keys()
            )
            query_mod = list(
                bm25s.tokenize(query, stopwords=None, stemmer=keyword_match_function)[1].keys()
            )
        if selected_search_method == "Oldest to newest":
            true_docs = run_boolean(
                query_mod, collection_full, selected_keyword_match_method
            )
            limited_docs = true_docs[:n]
            if len(limited_docs) > 0:
                print_results(collection_full,
                    limited_docs,
                    query_originals,
                    query_mod,
                    selected_keyword_match_method,
                )
            else:
                st.write("No results")
        elif selected_search_method == "Newest to oldest":
            true_docs = run_boolean(
                query_mod,
                collection_full,
                selected_keyword_match_method,
                order="reverse_chronological",
            )
            limited_docs = true_docs[:n]
            if len(limited_docs) > 0:
                print_results(collection_full,
                    limited_docs,
                    query_originals,
                    query_mod,
                    selected_keyword_match_method,
                )
            else:
                st.write("No results")
        elif selected_search_method == "Sparse vectors (BM25)":
            # Rank documents using sparse vectors (BM25)
            doc_score_map = run_query_sparse_vectors(
                query, collection_full, selected_keyword_match_method
            )
            top_docs = list(doc_score_map.keys())[:n]
            # For BM25, only keep if score above 0
            top_docs = [
                top_doc for top_doc in top_docs
                if doc_score_map[top_doc] > 0
            ]
            if len(top_docs) > 0:
                print_results(
                    collection_full,
                    top_docs,
                    query_originals,
                    query_mod,
                    selected_keyword_match_method,
                )
            else:
                st.write("No results")
        elif selected_search_method == "Dense vectors (embedding model)":
            # Rank documents using dense vectors
            doc_score_map = run_query_dense_vectors(
                query, collection_chunks
            )
            top_docs = list(doc_score_map.keys())[:n]
            print_results(
                collection_chunks,
                top_docs,
                query_originals,
                query_mod,
                selected_keyword_match_method,
            )
        elif selected_search_method == "Hybrid":
            map_sparse = run_query_sparse_vectors(
                query, collection_full, selected_keyword_match_method
            )
            map_dense = run_query_dense_vectors(query, collection_chunks)
            map_hybrid = run_hybrid(map_sparse.keys(), map_dense.keys())
            top_docs = list(map_hybrid.keys())[:n]
            print_results(
                collection_full,
                top_docs,
                query_originals,
                query_mod,
                selected_keyword_match_method,
            )
    elif query and not category_filter:
        st.write("Select at least one category")
    
    print(time.time() - start)

