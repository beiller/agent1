import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from typing import List, Dict, Optional
import json
import argparse
import subprocess

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

FASTEMBED_AVAILABLE = False
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    pass

# Use smallest, fastest model
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
MODEL = None
MODEL_LOCK = False

MAX_FILE_SIZE = 500_000

def is_text_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            return b'\x00' not in f.read(512)
    except:
        return False

def _ensure_model_loaded():
    global MODEL, MODEL_LOCK

    if MODEL:
        return MODEL
    
    if MODEL_LOCK:
        return None
        
    if not FASTEMBED_AVAILABLE:
        print("ERROR: fastembed not installed. Install with: pip install fastembed")
        return None
    
    MODEL_LOCK = True
    try:
        MODEL = TextEmbedding(model_name=MODEL_NAME)
        return MODEL
    except Exception as e:
        print(f"ERROR: Failed to load FastEmbed model: {e}")
        MODEL_LOCK = False
        return None

def create_store():
    return {
        'model': None,
        'chunks': [],
        'embeddings': None,
        'ids': [],
        'doc_positions': []
    }

def preprocess(text: str) -> str:
    return text

def add(store, text: str, doc_id: str, max_chunk_size: int = 300) -> None:
    if len(text) < 10:
        return
    
    if store['model'] is None:
        store['model'] = _ensure_model_loaded()
        if store['model'] is None:
            print("WARNING: Could not load FastEmbed model. Semantic search disabled.")
            return
    
    chunks = []
    positions = []
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        if len(para.strip()) <= max_chunk_size and len(para.strip()) > 20:
            chunks.append(para.strip())
            start = text.find(para.strip())
            positions.append((start, start + len(para.strip())))
        else:
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
            
            if current_chunk.strip() and len(current_chunk.strip()) > 20:
                chunks.append(current_chunk.strip())
                positions.append((chunk_start, chunk_start + len(current_chunk)))
    
    if not chunks:
        return
    
    try:
        # Use batch embedding for speed
        embeddings = list(store['model'].embed(chunks, batch_size=256))
        new_embeddings = np.array(embeddings)
        
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
    if store['model'] is None or store['embeddings'] is None:
        return []
    
    try:
        q_vec = np.array(list(store['model'].embed([query])))
        scores = cosine_similarity(q_vec, store['embeddings'])[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            if scores[i] > 0.3:
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
    text_files = []
    base_depth = directory.rstrip(os.sep).count(os.sep)
    
    for root, dirs, files in os.walk(directory):
        current_depth = root.count(os.sep) - base_depth
        
        if current_depth >= depth:
            dirs.clear()
            continue
        
        for filename in files:
            filepath = os.path.join(root, filename)
            
            try:
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    continue
            except:
                continue
            
            if not is_text_file(filepath):
                continue
            
            text_files.append(filepath)
    
    return sorted(text_files)

def pre_filter_documents(file_paths: List[str], query: str) -> List[str]:
    words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
    
    if not words:
        return file_paths
    
    # OR pattern: match if ANY word exists
    pattern = '|'.join(re.escape(w) for w in words)
    
    matching = []
    for fp in file_paths:
        result = subprocess.run(['grep', '-qiE', pattern, fp], 
                                capture_output=True)
        if result.returncode == 0:
            matching.append(fp)
    return matching

def build_database(directory: str = "conversations", depth: int = 8, 
                   pre_filter_query: Optional[str] = None, extension="") -> Dict:
    db = create_store()
    
    text_files = scan_directory(directory, depth)
    
    if not text_files:
        return db
    
    if pre_filter_query:
        matching_files = pre_filter_documents(text_files, pre_filter_query)
        files_to_process = matching_files
    else:
        files_to_process = text_files
    
    file_contents = {}
    for filepath in tqdm(files_to_process, desc="Loading files", disable=not HAS_TQDM):
        if not (extension != "" and filepath.lower().endswith(extension.lower())):
            continue
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            file_contents[filepath] = content
        except Exception as e:
            print(f"WARNING: Could not read {filepath}: {e}")
    
    for filepath in tqdm(files_to_process, desc="Processing", disable=not HAS_TQDM):
        if filepath in file_contents:
            content = file_contents[filepath]
            rel_path = os.path.relpath(filepath, directory)
            add(db, content, rel_path)
    
    print(f"Done! {len(db['chunks'])} chunks ready")
    return db

def search_database(db: Dict, keyword: str, context_window: int, top_k: int, 
                    top_j_per_doc: int = 3, base_dir: str = "conversations") -> str:
    all_results = []
    
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
    
    doc_groups = {}
    for r in all_results:
        filename = r['filename']
        if filename not in doc_groups:
            doc_groups[filename] = []
        doc_groups[filename].append(r)
    
    for filename in doc_groups:
        doc_groups[filename].sort(key=lambda x: x['score'], reverse=True)
        doc_groups[filename] = doc_groups[filename][:top_j_per_doc]
    
    doc_scores = [(filename, matches[0]['score']) for filename, matches in doc_groups.items()]
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top_docs = [doc[0] for doc in doc_scores[:top_k]]
    
    final_results = []
    for filename in top_docs:
        matches = doc_groups[filename]
        final_results.append({
            'filename': filename,
            'best_score': matches[0]['score'],
            'matches': matches
        })
    
    return json.dumps(final_results, indent=2)

def vector_search(
    keyword_to_search: str, context_size: int = 128, 
    top_k_docs: int = 3, top_j_per_doc: int = 2, 
    directory: str = "conversations", depth: int = 8,
    extension="") -> str:
    db = build_database(directory, depth, pre_filter_query=keyword_to_search, extension=extension)
    results = search_database(db, keyword_to_search, context_size, top_k_docs, top_j_per_doc, directory)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fast semantic search using FastEmbed.')
    
    parser.add_argument('keyword', type=str, help='Search query keyword or phrase')
    parser.add_argument('-d', '--directory', type=str, default='conversations',
                        help='Directory to scan (default: conversations)')
    parser.add_argument('--depth', type=int, default=8,
                        help='Maximum depth to traverse (default: 8)')
    parser.add_argument('-c', '--context-size', type=int, default=1000,
                        help='Characters of context (default: 1000)')
    parser.add_argument('-k', '--top-k', type=int, default=3,
                        help='Number of top documents (default: 3)')
    parser.add_argument('-j', '--top-j', type=int, default=2,
                        help='Number of matches per document (default: 2)')
    parser.add_argument('-e', '--extension', type=str, default="",
                        help='File extension to match')
    
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"ERROR: Directory '{args.directory}' does not exist.")
        exit(1)
    
    results = vector_search(
        keyword_to_search=args.keyword,
        context_size=args.context_size,
        top_k_docs=args.top_k,
        top_j_per_doc=args.top_j,
        directory=args.directory,
        depth=args.depth,
        extension=args.extension
    )
    
    print(results)
