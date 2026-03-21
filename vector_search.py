import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Dict, Tuple
import json

def create_store():
    return {
        'vectorizer': None, 
        'chunks': [],  
        'vectors': None, 
        'ids': [],  
        'doc_positions': []
    }

def preprocess(text: str) -> str: 
    return re.sub(r'[^\w\u2000-\u3300\ud800-\udfff]+', ' ', text).replace("  ", " ").replace("  ", " ").replace("  ", " ")

def add(store, text, doc_id, max_chunk_size=300):
    """Add document to store using paragraph-aware chunking"""
    if len(text) < 10:
        return

    chunks = []
    positions = []
    
    # Split by paragraphs first (double newlines)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If paragraph is already small enough, use it as-is
        if len(para.strip()) <= max_chunk_size and len(para.strip()) > 20:
            chunks.append(para.strip())
            # Find position in original text (approximate)
            start = text.find(para.strip())
            positions.append((start, start + len(para.strip())))
        else:
            # Split large paragraphs by sentences or lines
            current_chunk = ""
            chunk_start = 0
            
            for line in para.split('\n'):
                if len(current_chunk) + len(line) > max_chunk_size:
                    if current_chunk.strip() and len(current_chunk.strip()) > 20:
                        chunks.append(current_chunk.strip())
                        positions.append((chunk_start, chunk_start + len(current_chunk)))
                    current_chunk = line
                    chunk_start = para.find(line, chunk_start)
                else:
                    current_chunk += "\n" + line
            
            # Don't forget the last chunk
            if current_chunk.strip() and len(current_chunk.strip()) > 20:
                chunks.append(current_chunk.strip())
                positions.append((chunk_start, chunk_start + len(current_chunk)))
    
    if not chunks:
        return
    
    # BATCH vectorization
    if store['vectorizer'] is None:
        store['vectorizer'] = TfidfVectorizer(
            analyzer='word', 
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        store['vectors'] = store['vectorizer'].fit_transform(chunks).toarray()
    else:
        new_vectors = store['vectorizer'].transform(chunks).toarray()
        store['vectors'] = np.vstack([store['vectors'], new_vectors])
    
    chunk_idx_start = len(store['chunks'])
    store['chunks'].extend(chunks)
    store['ids'].extend([(doc_id, i) for i in range(chunk_idx_start, chunk_idx_start + len(chunks))])
    store['doc_positions'].extend(positions)

def search(store, query, top_k):
    """Search for matching chunks across all documents"""
    if store['vectorizer'] is None:
        return []
    
    q_vec = store['vectorizer'].transform([query]).toarray()
    scores = cosine_similarity(q_vec, store['vectors'])[0]
    
    # PENALIZE CHUNKS WITH EXCESSIVE BACKSLASHES (simple, memory-efficient)
    for i in range(len(store['chunks'])):
        chunk = store['chunks'][i]
        slash_count = chunk.count('\\')
        total_chars = max(len(chunk), 1)
        
        # If >5% of characters are backslashes, penalize heavily
        scores[i] *= 1 / (slash_count+1)  # Reduce score by 90%
    
    # Get top_k indices directly using argsort
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        if scores[i] > 0.1:
            results.append({
                'id': store['ids'][i],
                'chunk': store['chunks'][i],
                'score': float(scores[i]),
                'doc_start': store['doc_positions'][i][0],
                'doc_end': store['doc_positions'][i][1]
            })
    
    return results

def build_database():
    """Build database from all txt files in conversations directory"""
    db = create_store()
    
    conversations_dir = "conversations"
    if os.path.exists(conversations_dir):
        for filename in sorted(os.listdir(conversations_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(conversations_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                add(db, content, filename)
    
    return db

def search_database(db, keyword: str, context_window: int, top_k: int, top_j_per_doc: int = 3):  
    """Search database and return top_k documents with top_j matches per document"""
    all_results = []
    
    # Get more candidates to ensure we have enough after grouping by document
    num_chunks_to_fetch = top_k * 10  # Fetch extra chunks to get more document coverage
    for result in search(db, keyword, top_k=num_chunks_to_fetch):
        filename = result['id'][0]
        
        filepath = os.path.join("conversations", filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            full_doc = preprocess(f.read())
        
        start = max(0, result['doc_start'] - context_window)
        end = min(len(full_doc), result['doc_end'] + context_window)
        snippet = full_doc[start:end]
        
        all_results.append({
            'filename': filename,
            'snippet': snippet,
            'score': result['score'],
            'chunk_id': result['id'][1],
            'doc_start': result['doc_start'],
            'doc_end': result['doc_end']
        })
    
    # Group results by document (filename)
    doc_groups = {}
    for r in all_results:
        filename = r['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(r)
    
    # Sort each document's matches by score and keep top_j_per_doc
    for filename in doc_groups:
        doc_groups[filename].sort(key=lambda x: x['score'], reverse=True)
        doc_groups[filename] = doc_groups[filename][:top_j_per_doc]
    
    # Calculate best score per document (for ranking documents)
    doc_scores = [(filename, matches[0]['score']) for filename, matches in doc_groups.items()]
    
    # Sort documents by their best match score and keep top_k documents
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc[0] for doc in doc_scores[:top_k]]
    
    # Build final result structure
    final_results = []
    for filename in top_docs:
        matches = doc_groups[filename]
        final_results.append({
            'filename': filename,
            'best_score': matches[0]['score'],
            'matches': matches  # List of top_j_per_doc matches within this document
        })
    
    return json.dumps(final_results, indent=True)

def vector_search(keyword_to_search: str, context_size: int = 1000, top_k: int = 3, top_j_per_doc: int = 2) -> str:
    """Search for previous conversation history about a conversation topic. 
    
    Provide a `keyword_to_search`. This is a string that represents the conversation words that may be in chat history. Only provide words that will contextually fit what the user is asking.

    Optionally use `context_size` if you need more information and you were not successfull at first. Use a larger size up to 1000+
    
    This function returns JSON with top matching snippets from documents, 
    ranked by relevance score within each document.
    
    Args:
        keyword_to_search: The search query
        context_size: How many characters of context to include around matches
        top_k: Number of top documents to return (not per-document) (default: 3)
        top_j_per_doc: Number of top matches to show within each document (default: 2)
    """
    db = build_database()
    results = search_database(db, keyword_to_search, context_size, top_k, top_j_per_doc)
    return results


if __name__ == "__main__":
    import sys

    print(vector_search(sys.argv[1]))
