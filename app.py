"""Streamlit app for searching transcripts and newsletters pulling from
ChromaDB collections. Can choose between sparse vectors (BM25), dense
vectors (embedding model), or hybrid.
"""

import json
import time
from itertools import chain
from operator import itemgetter

import bm25s
import chromadb
import Stemmer
import streamlit as st
from bm25s.tokenization import Tokenized
from chromadb.api.models.Collection import Collection


def run_query_sparse_vectors(
        query_tokenized: Tokenized,
        collection: Collection,
    ) -> dict[str, float]:
    """Run query and rank document relevance using sparse vectors.

    Args:
        query_tokenized: tokenized query object (tokenize beforehand so we only
                         have to do it once)
        collection: chromadb collection
    Returns:
        doc_score_map: mapping from title of document to score
    """
    collection_filtered = collection.get(where=category_filter)
    corpus = collection_filtered["documents"]
    titles = [
        item["title"]
        for item in collection_filtered["metadatas"]
    ]
    doc_title_map = dict(zip(corpus, titles))
    # Default parameter values are k1=1.5, b=0.75
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus, stemmer=stemmer))
    num_docs = len(corpus)
    docs_ordered, scores = retriever.retrieve(
        query_tokenized, k=num_docs
    )
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


def write_keyword_highlighting(
        collection: Collection,
        target_id: str,
        query_tokenized: Tokenized,
    ) -> None:
    """Write snippets of document with keywords highlighted.

    Args:
        collection: chromadb collection
        target_id: id of document/chunk that we're displaying keywords for
        query_tokenized: tokenized query object (tokenize beforehand so we only
                         have to do it once)
    """
    id_collection = collection.get(ids=[target_id])
    doc = id_collection["documents"][0].strip()
    doc_pieces = doc.split(" ")
    # Mapping of doc piece to not stemmed tokens
    doc_piece_map = json.loads(
        id_collection["metadatas"][0]["doc_piece_map"]
    )
    # Mapping of stemmed tokens to not stemmed tokens
    token_map = json.loads(
        id_collection["metadatas"][0]["token_map"]
    )
    # Keep only query tokens that are in document
    query_tokens = set(query_tokenized[1].keys())
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
        doc_score_map: dict[str, float],
        query_tokenized: Tokenized,
    ) -> None:
    """Print top search results.

    Includes title, link, score, and text with keywords from query highlighted.

    Args:
        collection: chromadb collection
        top_docs: list of titles of top ranking documents
        doc_score_map: mapping of document title to score
        query_tokenized: tokenized query object (tokenize beforehand so we only
                         have to do it once)
    """
    for top_doc in top_docs:
        # We won't need category_filter here because we've already narrowed down
        # the documents
        collection_subset = collection.get(where={"title": top_doc})
        metadata_temp = collection_subset["metadatas"][0]
        # Write results
        st.markdown(f"##### [{metadata_temp['title']}]({metadata_temp['link']})")
        with st.expander("Keyword matches"):
            st.write(f"Score: {doc_score_map[top_doc]}")
            ids = collection_subset["ids"]
            for target_id in ids:
                write_keyword_highlighting(collection, target_id, query_tokenized)


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
    st.title("Information retrieval for MAIHT3k")
    
    # Define options
    with st.sidebar:
        st.sidebar.header("Options")
        n = int(st.text_input("Number of results", "5"))
        cat_options = ["Podcast transcripts", "Newsletters"]
        selected_cat_options = st.multiselect(
            "Select one or more categories:",
            cat_options,
            default=cat_options,
        )
        search_methods = [
            "Sparse vectors (BM25)", "Dense vectors (embedding model)", "Hybrid"
        ]
        selected_search_method = st.selectbox(
            "Select search method:", search_methods
        )
        if set(selected_cat_options) == set(cat_options):
            category_filter = {
                "$or": [{"category": "transcripts"}, {"category": "newsletters"}]
            }
        elif selected_cat_options == ["Podcast transcripts"]:
            category_filter = {"category": "transcripts"}
        elif selected_cat_options == ["Newsletters"]:
            category_filter = {"category": "newsletters"}

    # Run query
    query = st.text_input("Enter search query", "")
    if query and category_filter:
        query_tokenized = bm25s.tokenize(query, stemmer=stemmer)
        if selected_search_method == "Sparse vectors (BM25)":
            # Rank documents using sparse vectors (BM25)
            doc_score_map = run_query_sparse_vectors(
                query_tokenized, collection_full
            )
            top_docs = list(doc_score_map.keys())[:n]
            # For BM25, only keep if score above 0
            top_docs = [
                top_doc for top_doc in top_docs
                if doc_score_map[top_doc] > 0
            ]
            if len(top_docs) > 0:
                print_results(collection_full, top_docs, doc_score_map, query_tokenized)
            else:
                st.write("No results")
        elif selected_search_method == "Dense vectors (embedding model)":
            # Rank documents using dense vectors
            doc_score_map = run_query_dense_vectors(
                query, collection_chunks
            )
            top_docs = list(doc_score_map.keys())[:n]
            print_results(collection_chunks, top_docs, doc_score_map, query_tokenized)
        elif selected_search_method == "Hybrid":
            map_sparse = run_query_sparse_vectors(query_tokenized, collection_full)
            map_dense = run_query_dense_vectors(query, collection_chunks)
            map_hybrid = run_hybrid(
                map_sparse.keys(), map_dense.keys()
            )
            top_docs = list(map_hybrid.keys())[:n]
            print_results(collection_full, top_docs, map_hybrid, query_tokenized)
    elif query and not category_filter:
        st.write("Select at least one category")
    
    print(time.time() - start)

