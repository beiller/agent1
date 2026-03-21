import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Dict, Optional
import json
import hashlib
import pickle

# Lazy import - will fail gracefully if not installed
SENTENCE_TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, small (~80MB), good quality
MODEL_CACHE_PATH = os.path.expanduser('~/.cache/agent1/sentence_transformer')

def _ensure_model_loaded() -> Optional[SentenceTransformer]:
    """Load or download the sentence transformer model. Auto-downloads on first use."""
    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("ERROR: sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    
    # Create cache directory
    os.makedirs(MODEL_CACHE_PATH, exist_ok=True)
    
    try:
        # This will download the model on first use and cache it
        model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_PATH, device="cpu")
        return model
    except Exception as e:
        print(f"ERROR: Failed to load sentence transformer model: {e}")
        return None

def create_store():
    """Create a new vector store for semantic search."""
    return {
        'model': None,  # SentenceTransformer model (loaded lazily)
        'chunks': [],   # List of text chunks
        'embeddings': None,  # numpy array of embeddings
        'ids': [],      # List of (doc_id, chunk_index) tuples
        'doc_positions': []  # List of (start, end) positions in original doc
    }

def preprocess(text: str) -> str:
    """Clean text for processing."""
    return re.sub(r'[^\w\u2000-\u3300\ud800-\udfff]+', ' ', text).replace("  ", " ").strip()

def add(store, text: str, doc_id: str, max_chunk_size: int = 300) -> None:
    """Add document to store using paragraph-aware chunking."""
    if len(text) < 10:
        return
    
    # Load model on first use
    if store['model'] is None:
        store['model'] = _ensure_model_loaded()
        if store['model'] is None:
            print("WARNING: Could not load sentence transformer model. Semantic search disabled.")
            return
    
    chunks = []
    positions = []
    
    # Split by paragraphs first (double newlines)
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If paragraph is already small enough, use it as-is
        if len(para.strip()) <= max_chunk_size and len(para.strip()) > 20:
            chunks.append(para.strip())
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
    
    # Generate embeddings for new chunks
    try:
        new_embeddings = store['model'].encode(chunks, show_progress_bar=False)
        
        if store['embeddings'] is None:
            store['embeddings'] = new_embeddings
        else:
            store['embeddings'] = np.vstack([store['embeddings'], new_embeddings])
    except Exception as e:
        print(f"ERROR: Failed to generate embeddings: {e}")
        return
    
    chunk_idx_start = len(store['chunks'])
    store['chunks'].extend(chunks)
    store['ids'].extend([(doc_id, i) for i in range(chunk_idx_start, chunk_idx_start + len(chunks))])
    store['doc_positions'].extend(positions)

def search(store, query: str, top_k: int) -> List[Dict]:
    """Search for matching chunks using semantic similarity."""
    if store['model'] is None or store['embeddings'] is None:
        return []
    
    try:
        # Generate embedding for query
        q_vec = store['model'].encode([query], show_progress_bar=False)
        
        # Calculate cosine similarity
        scores = cosine_similarity(q_vec, store['embeddings'])[0]
        
        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0.3:  # Semantic similarity threshold (0-1 scale)
                results.append({
                    'id': store['ids'][i],
                    'chunk': store['chunks'][i],
                    'score': float(scores[i]),
                    'doc_start': store['doc_positions'][i][0],
                    'doc_end': store['doc_positions'][i][1]
                })
        
        return results
    except Exception as e:
        print(f"ERROR: Search failed: {e}")
        return []

def build_database() -> Dict:
    """Build database from all txt files in conversations directory."""
    db = create_store()
    
    conversations_dir = "conversations"
    if os.path.exists(conversations_dir):
        for filename in sorted(os.listdir(conversations_dir)):
            if filename.endswith(".txt"):
                filepath = os.path.join(conversations_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    add(db, content, filename)
                except Exception as e:
                    print(f"WARNING: Could not read {filename}: {e}")
    
    return db

def search_database(db: Dict, keyword: str, context_window: int, top_k: int, top_j_per_doc: int = 3) -> str:
    """Search database and return top_k documents with top_j matches per document."""
    all_results = []
    
    # Get more candidates to ensure we have enough after grouping by document
    num_chunks_to_fetch = top_k * 10
    for result in search(db, keyword, top_k=num_chunks_to_fetch):
        filename = result['id'][0]
        
        filepath = os.path.join("conversations", filename)
        try:
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
        except Exception as e:
            print(f"WARNING: Could not process {filename}: {e}")
    
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
    
    return json.dumps(final_results, indent=2)

def vector_search(keyword_to_search: str, context_size: int = 1000, top_k: int = 3, top_j_per_doc: int = 2) -> str:
    """Search for previous conversation history about a conversation topic. 
    
    Provide a `keyword_to_search`. This is a string that represents the conversation words 
    that may be in chat history. Only provide words that will contextually fit what the user is asking.

    Optionally use `context_size` if you need more information and you were not successful at first. 
    Use a larger size up to 1000+
    
    This function returns JSON with top matching snippets from documents, 
    ranked by relevance score within each document.
    
    The search uses semantic embeddings (sentence-transformers) which understand meaning,
    not just keywords. First run will download the model (~80MB), subsequent runs are cached.
    
    Args:
        keyword_to_search: The search query
        context_size: How many characters of context to include around matches
        top_k: Number of top documents to return (not per-document) (default: 3)
        top_j_per_doc: Number of top matches to show within each document (default: 2)
    
    Returns:
        JSON string with search results
    """
    db = build_database()
    results = search_database(db, keyword_to_search, context_size, top_k, top_j_per_doc)
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_search.py <keyword>")
        sys.exit(1)
    
    print(vector_search(sys.argv[1]))
