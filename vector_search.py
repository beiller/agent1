import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Dict, Optional
import json
import argparse
import glob

# Lazy import - will fail gracefully if not installed
SENTENCE_TRANSFORMER_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    pass

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'  # Fast, small (~80MB), good quality
MODEL_CACHE_PATH = '.cache'
MODEL = None

def _ensure_model_loaded() -> Optional[SentenceTransformer]:
    """Load or download the sentence transformer model. Auto-downloads on first use."""
    global MODEL

    if MODEL: return MODEL

    if not SENTENCE_TRANSFORMER_AVAILABLE:
        print("ERROR: sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    
    # Create cache directory
    use_local = os.path.isdir(".cache")
    if not use_local:
        os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

    try:
        # This will download the model on first use and cache it
        MODEL = SentenceTransformer(
            MODEL_NAME, 
            cache_folder=MODEL_CACHE_PATH, 
            device="cpu",
            local_files_only=use_local
        )
        return MODEL
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
    return re.sub(r'[\W_]+', ' ', text).replace("  ", " ").strip()

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

def scan_directory(directory: str, depth: int = 8) -> List[str]:
    """Recursively scan directory for text files up to specified depth.
    
    Args:
        directory: Root directory to scan
        depth: Maximum depth to traverse (default: 8)
    
    Returns:
        List of absolute file paths to .txt files
    """
    txt_files = []
    base_depth = directory.rstrip(os.sep).count(os.sep)
    
    for root, dirs, files in os.walk(directory):
        # Calculate current depth relative to base directory
        current_depth = root.count(os.sep) - base_depth
        
        # Stop if we've exceeded the maximum depth
        if current_depth >= depth:
            dirs.clear()  # Don't descend further into subdirectories
            continue
        
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                txt_files.append(filepath)
    
    return sorted(txt_files)

def build_database(directory: str = "conversations", depth: int = 8) -> Dict:
    """Build database from all txt files in specified directory.
    
    Args:
        directory: Root directory to scan for .txt files
        depth: Maximum depth to traverse down the directory tree (default: 8)
    
    Returns:
        Vector store dictionary
    """
    db = create_store()
    
    # Scan directory for all .txt files up to specified depth
    txt_files = scan_directory(directory, depth)
    
    print(f"Found {len(txt_files)} text files in '{directory}' (depth={depth})")
    
    for filepath in txt_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use relative path from directory as doc_id for cleaner output
            rel_path = os.path.relpath(filepath, directory)
            add(db, content, rel_path)
        except Exception as e:
            print(f"WARNING: Could not read {filepath}: {e}")
    
    return db

def search_database(db: Dict, keyword: str, context_window: int, top_k: int, top_j_per_doc: int = 3, base_dir: str = "conversations") -> str:
    """Search database and return top_k documents with top_j matches per document.
    
    Args:
        db: Vector store dictionary
        keyword: Search query string
        context_window: Characters of context to include around each match
        top_k: Number of top documents to return
        top_j_per_doc: Number of top matches per document (default: 3)
        base_dir: Base directory for resolving file paths
    
    Returns:
        JSON string with search results
    """
    all_results = []
    
    # Get more candidates to ensure we have enough after grouping by document
    num_chunks_to_fetch = top_k * 10
    for result in search(db, keyword, top_k=num_chunks_to_fetch):
        rel_path = result['id'][0]
        
        filepath = os.path.join(base_dir, rel_path)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                full_doc = preprocess(f.read())
            
            start = max(0, result['doc_start'] - context_window)
            end = min(len(full_doc), result['doc_end'] + context_window)
            snippet = full_doc[start:end]
            
            all_results.append({
                'filename': rel_path,
                'snippet': snippet,
                'score': result['score'],
                'chunk_id': result['id'][1],
                'doc_start': result['doc_start'],
                'doc_end': result['doc_end']
            })
        except Exception as e:
            print(f"WARNING: Could not process {filepath}: {e}")
    
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

def vector_search(keyword_to_search: str, context_size: int = 1000, top_k_docs: int = 3, 
                  top_j_per_doc: int = 2, directory: str = "conversations", depth: int = 8) -> str:
    """Search for keyword matches in text, code, anything! Returns surrounding context
    
    Provide a `keyword_to_search`. This is a string that represents the conversation words 
    that may be in chat history.

    Optionally use `context_size` if you need more information and you were not successful at first. 
    Use a larger size up to 1000+
    
    This function returns JSON with top matching snippets from documents, 
    ranked by relevance score within each document.
    
    The search uses semantic embeddings (sentence-transformers) which understand meaning,
    not just keywords.
    
    Args:
        keyword_to_search: The search query
        context_size: How many characters of context to include around matches
        top_k_docs: Number of top documents to return (not per-document)
        top_j_per_doc: Number of top matches to show within each document
        directory: Directory to scan for .txt files (default: "conversations", feel free to contextually search even code!)
        depth: Maximum depth to traverse down the directory tree (default: 8)

    Returns:
        JSON string with search results
    """
    db = build_database(directory, depth)
    results = search_database(db, keyword_to_search, context_size, top_k_docs, top_j_per_doc, directory)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Powerful semantic search tool for text files using sentence embeddings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python vector_search.py "conversation topic"                    # Search in ./conversations (depth=8)
  python vector_search.py "topic" -d ./my_docs                   # Search in custom directory
  python vector_search.py "topic" -d . --depth 3                 # Search current dir, max depth 3
  python vector_search.py "topic" -c 2000 -k 5 -j 4              # Custom context and result counts
        '''
    )
    
    parser.add_argument('keyword', type=str, help='Search query keyword or phrase')
    parser.add_argument('-d', '--directory', type=str, default='conversations',
                        help='Directory to scan for .txt files (default: conversations)')
    parser.add_argument('--depth', type=int, default=8,
                        help='Maximum depth to traverse down directory tree (default: 8)')
    parser.add_argument('-c', '--context-size', type=int, default=1000,
                        help='Characters of context around each match (default: 1000)')
    parser.add_argument('-k', '--top-k', type=int, default=3,
                        help='Number of top documents to return (default: 3)')
    parser.add_argument('-j', '--top-j', type=int, default=2,
                        help='Number of top matches per document (default: 2)')
    
    args = parser.parse_args()
    
    # Validate directory exists
    if not os.path.isdir(args.directory):
        print(f"ERROR: Directory '{args.directory}' does not exist.")
        exit(1)
    
    # Run search
    results = vector_search(
        keyword_to_search=args.keyword,
        context_size=args.context_size,
        top_k_docs=args.top_k,
        top_j_per_doc=args.top_j,
        directory=args.directory,
        depth=args.depth
    )
    
    print(results)

