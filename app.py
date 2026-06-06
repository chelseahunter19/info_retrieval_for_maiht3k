

import time

import bm25s
import chromadb
import Stemmer
import streamlit as st


def run_query_dense_embeddings(query, collection):
    """Run query using dense embeddings

    query: str
    collection: chromadb collection
    n_results: int
    """

    results = collection.query(
        query_texts = [query],
        n_results = collection.count()
    )
    ids_ordered = results["ids"][0]
    return ids_ordered

def run_query_sparse_embeddings(query, collection):
    """Run query using sparse embeddings

    query: str
    collection: chromadb collection
    n_results: int

    Returns:
    ids in order of score
    """

    corpus = collection.get()["documents"]
    #print(len(corpus))
    #print(len(list(set(corpus))))
    
    ## Check out vocab
    #corpus_tokens = bm25s.tokenize(corpus, stopwords=None, stemmer=stemmer)
    #positions = corpus_tokens[0][0]
    #vocab_mapping_reversed = {corpus_tokens[1][key]: key for key in corpus_tokens[1]}
    #for position in positions:
    #    print(vocab_mapping_reversed[position])

    ## IDs first element, vocab second element
    #for key in corpus_tokens[1]:
    #    print(key, corpus_tokens[1][key])

    # Map back to ids from collection object
    mapping = dict(zip(collection.get()["ids"], corpus))
    mapping_inverted = {mapping[key]: key for key in mapping}

    #from collections import Counter
    #counts = Counter(corpus)
    #duplicates = [item for item, count in counts.items() if count > 1]
    #print(len(duplicates))
    
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(bm25s.tokenize(corpus, stemmer=stemmer))
    
    docs_ordered, scores = retriever.retrieve(bm25s.tokenize(query, stemmer=stemmer), k=collection.count())
    ids_ordered = [mapping_inverted[doc] for doc in docs_ordered[0]]

    return ids_ordered, scores[0]

def run_hybrid(ids_dense, ids_sparse):
    """Run hybrid search ranking
    """

    # rank dictionary: mapping of id to rank number
    rank_dense = dict(zip(ids_dense, range(len(ids_dense))))
    rank_sparse = dict(zip(ids_sparse, range(len(ids_sparse))))

    hybrid_scores = {}
    k = 60
    w_dense = 1
    w_sparse = 1
    for key in rank_dense:
        hybrid_scores[key] = (w_dense/(k+rank_dense[key])) + (w_sparse/(k+rank_sparse[key]))
    
    hybrid_scores_sorted = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
    hybrid_ids_ordered = [item[0] for item in hybrid_scores_sorted]
    return hybrid_ids_ordered

def write_keyword_highlighting(target_id, results, id_order_mapping):
    """Write snippets of document with keywords highlighted

    target_id: str
    results: dict
    id_order_mapping: dict
    """

    doc = results["documents"][id_order_mapping[target_id]].strip()
    doc_tokens = bm25s.tokenize(doc, stemmer=stemmer)[1].keys()
    query_tokens = bm25s.tokenize(query, stemmer=stemmer)[1].keys()

    doc_piece_mapping = {doc_piece: list(bm25s.tokenize(doc_piece, stopwords=None, stemmer=stemmer)[1]) for doc_piece in set(doc.split(" "))}
    query_token_positions = [i for i, item in enumerate(doc.split(" ")) if set(doc_piece_mapping[item]).intersection(set(query_tokens))]
    positions_grouped = []
    span_size = 10
    for i, position in enumerate(query_token_positions):
        if i == 0:
            positions_grouped.append([position])
        else:
            if position <= (query_token_positions[i-1] + span_size):
                positions_grouped[-1].append(position)
            else:
                positions_grouped.append([position])
    for position_group in positions_grouped:
        start = max(0, position_group[0]-span_size)
        end = min(position_group[-1]+span_size, len(doc.split(" "))-1)
        span_items = doc.split(" ")[start:end+1]
        
        for item in span_items:
            temp = list(bm25s.tokenize(item, stopwords=None, stemmer=stemmer)[1])
        span_items_w_emphasis = []
        for item in span_items:
            # Need to not stem to match original text
            item_tokenized = bm25s.tokenize(item, stopwords=None, stemmer=None, lower=False)
            vocab = item_tokenized[1]
            if vocab:
                for token in list(vocab):
                    # Need to stem now to match query tokens
                    if list(bm25s.tokenize(token, stopwords=None, stemmer=stemmer)[1])[0] in query_tokens:
                        item = item.replace(token, f"<mark><strong>{token}</mark></strong>")
            span_items_w_emphasis.append(item)
        span = " ".join(span_items_w_emphasis)
        st.html(span)


if __name__ == "__main__":

    start = time.time()
    stemmer = Stemmer.Stemmer("english")
    collection = None
    client = chromadb.PersistentClient(path="chroma_db")
    
    st.title("Information retrieval for MAIHT3k")
    
    # Define options
    with st.sidebar:
        st.sidebar.header("Options")
        n = int(st.text_input("Number of results", "5"))
        selected_category_options = st.multiselect("Select one or more categories:", ["Podcast transcripts", "Newsletters"],
                                                 default = ["Podcast transcripts", "Newsletters"])
        if set(selected_category_options) == set(["Podcast transcripts", "Newsletters"]):
            collection = client.get_collection(name="maiht3k_all")
        elif selected_category_options == ["Podcast transcripts"]:
            collection = client.get_collection(name="maiht3k_transcripts")
        elif selected_category_options == ["Newsletters"]:
            collection = client.get_collection(name="maiht3k_newsletters")
    
    query = st.text_input("Enter search query", "")
    
    if query and collection:
        # Calculate BM25 scores
        ids_sparse, scores_sparse = run_query_sparse_embeddings(query, collection)
        scores_mapping = {ids_sparse[i]: scores_sparse[i] for i in range(len(ids_sparse))}
        
        top_ids = ids_sparse[:n]
        # Only keep if score above 0
        top_ids = [top_id for top_id in top_ids if scores_mapping[top_id] > 0]
        if len(top_ids) > 0:
            top_n_sparse_results = collection.get(ids=top_ids)
            id_order_mapping = {top_n_sparse_results["ids"][i]: i for i in range(len(top_n_sparse_results["ids"]))}
    
            for target_id in top_ids:
                metadata_temp = top_n_sparse_results["metadatas"][id_order_mapping[target_id]]
                # Write results
                st.header(metadata_temp["title"])
                st.write(metadata_temp["link"])
                st.write(f"Score: {scores_mapping[target_id]}")
    
                # Keyword highlighting
                write_keyword_highlighting(target_id, top_n_sparse_results, id_order_mapping)
        else:
            st.write("No results")
    elif query and not collection:
        st.write("Select at least one category")
    
    
    print(time.time()-start)

