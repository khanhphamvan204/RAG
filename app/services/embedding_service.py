# app/services/embedding_service.py
import os
import logging
import gc
from typing import List, Optional
from functools import lru_cache
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from app.config import Config
from app.services.file_service import get_file_paths
from app.services.metadata_service import find_document_info
from app.services.document_loader import load_new_documents

# Lựa chọn 1: HuggingFace Embeddings (nhẹ nhất)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Lựa chọn 2: OpenAI Embeddings (cần API key)
# from langchain_community.embeddings import OpenAIEmbeddings

# Lựa chọn 3: Ollama Embeddings (local, nhẹ)
# from langchain_community.embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

# Global embedding model cache
_embedding_model_cache = None
_embedding_model_lock = None

try:
    import threading
    _embedding_model_lock = threading.Lock()
except ImportError:
    # Fallback nếu không có threading
    class DummyLock:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    _embedding_model_lock = DummyLock()

@lru_cache(maxsize=1)
def _create_embedding_model():
    """Private function to create embedding model with caching"""
    logger.info("Creating new embedding model instance...")
    
    # Option 1: HuggingFace Embeddings (nhẹ nhất, không cần GPU)
    model = HuggingFaceEmbeddings(
        model_name="dangvantuan/vietnamese-document-embedding",
        model_kwargs={'device': 'cpu','trust_remote_code':True}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Option 2: OpenAI Embeddings (nếu có API key)
    # model = OpenAIEmbeddings(
    #     model="text-embedding-3-small",  # Nhẹ và rẻ
    #     openai_api_key=os.getenv("OPENAI_API_KEY")
    # )
    
    # Option 3: Ollama Embeddings (local, rất nhẹ)
    # model = OllamaEmbeddings(
    #     model="nomic-embed-text:latest",
    #     base_url="http://localhost:11434"
    # )
    
    logger.info("Embedding model created successfully")
    return model

def get_embedding_model():
    """Get consistent embedding model with singleton pattern"""
    global _embedding_model_cache
    
    if _embedding_model_cache is None:
        with _embedding_model_lock:
            # Double-check pattern
            if _embedding_model_cache is None:
                _embedding_model_cache = _create_embedding_model()
    
    return _embedding_model_cache

def clear_embedding_model_cache():
    """Clear embedding model cache - useful for memory management"""
    global _embedding_model_cache
    with _embedding_model_lock:
        if _embedding_model_cache is not None:
            logger.info("Clearing embedding model cache")
            _embedding_model_cache = None
            # Clear LRU cache
            _create_embedding_model.cache_clear()
            gc.collect()

class EmbeddingModelManager:
    """Context manager for embedding model to ensure cleanup"""
    def __init__(self):
        self.model = None
    
    def __enter__(self):
        self.model = get_embedding_model()
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Có thể thêm cleanup logic nếu cần
        pass

def semantic_sliding_window_split(text: str, embedding_model, window_overlap: float = 0.2) -> List[str]:
    """
    Sliding window với tỷ lệ overlap dựa trên semantic boundaries
    
    Args:
        text: Văn bản cần chia
        embedding_model: Model embedding (đã được cached)
        window_overlap: Tỷ lệ overlap (0.0-1.0)
    
    Returns:
        List các chunk có overlap theo tỷ lệ
    """
    try:
        semantic_chunker = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            # sentence_split_regex=r'(?<!\b[A-Za-zÀ-ỹ].)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ].)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ][A-Za-zÀ-ỹ].)(?<=[.?!…])\s+(?=[A-ZÀ-Ỵ0-9])',
            sentence_split_regex=r'(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<!\b[A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<!\b[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ][A-Za-zÀ-ỹ]\.)(?<=[.?!…])\s+(?=[A-ZÀ-Ỵ0-9])',
            buffer_size=5,
        )
        
        chunks = semantic_chunker.split_text(text)
        
        if len(chunks) <= 1:
            return chunks
        
        sliding_chunks = []
        
        for i in range(len(chunks)):
            if i == 0:
                sliding_chunks.append(chunks[i])
            else:
                # Tính toán overlap size dựa trên tỷ lệ
                prev_chunk_words = chunks[i-1].split()
                overlap_words_count = int(len(prev_chunk_words) * window_overlap)
                
                if overlap_words_count > 0:
                    # Lấy overlap theo từ thay vì ký tự để tự nhiên hơn
                    overlap_text = ' '.join(prev_chunk_words[-overlap_words_count:])
                    new_chunk = overlap_text + " " + chunks[i]
                else:
                    new_chunk = chunks[i]
                
                sliding_chunks.append(new_chunk)
        
        return sliding_chunks
        
    except Exception as e:
        logger.warning(f"Semantic sliding window failed: {e}")
        return [text]  # Fallback: trả về text gốc

def get_text_splitter(use_semantic: bool = True, semantic_overlap: float = 0.2, embedding_model=None):
    """Get text splitter - semantic or traditional"""
    try:
        if use_semantic:
            logger.info("Using SemanticChunker with sliding window overlap for text splitting")
            
            # Sử dụng model đã có sẵn thay vì load mới
            if embedding_model is None:
                embedding_model = get_embedding_model()
            
            # Custom wrapper class để tích hợp semantic sliding window
            class SemanticSlidingWindowSplitter:
                def __init__(self, embedding_model, window_overlap=0.2):
                    self.embedding_model = embedding_model
                    self.window_overlap = window_overlap
                
                def split_text(self, text: str) -> List[str]:
                    return semantic_sliding_window_split(text, self.embedding_model, self.window_overlap)
                
                def split_documents(self, documents: List[Document]) -> List[Document]:
                    chunks = []
                    for doc in documents:
                        text_chunks = self.split_text(doc.page_content)
                        for chunk_text in text_chunks:
                            chunk_doc = Document(
                                page_content=chunk_text,
                                metadata=doc.metadata.copy()
                            )
                            chunks.append(chunk_doc)
                    return chunks
            
            return SemanticSlidingWindowSplitter(embedding_model, semantic_overlap)
        else:
            logger.info("Using RecursiveCharacterTextSplitter for text splitting")
            return RecursiveCharacterTextSplitter(
                chunk_size=2000, 
                chunk_overlap=200
            )
    except Exception as e:
        logger.warning(f"Failed to create SemanticSlidingWindowSplitter: {e}")
        logger.info("Falling back to RecursiveCharacterTextSplitter")
        return RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=200
        )

def add_to_embedding(file_path: str, metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2):
    """Add documents to vector database - optimized version"""
    try:
        logger.info(f"Starting embedding process for: {file_path}")
        
        # Load documents
        documents = load_new_documents(file_path, metadata)
        if not documents:
            logger.warning(f"No documents loaded from {file_path}")
            return False
        logger.info(documents);

        # Get embedding model once và dùng chung
        with EmbeddingModelManager() as embedding_model:
            # Split into chunks with semantic or traditional splitter
            text_splitter = get_text_splitter(
                use_semantic=use_semantic_chunking, 
                semantic_overlap=semantic_overlap,
                embedding_model=embedding_model  # Pass model đã có
            )
            
            try:
                chunks = text_splitter.split_documents(documents)
                if use_semantic_chunking:
                    logger.info(f"Successfully created {len(chunks)} chunks using semantic chunking with {semantic_overlap*100}% overlap")
                else:
                    logger.info(f"Successfully created {len(chunks)} chunks using traditional chunking")
            except Exception as e:
                if use_semantic_chunking:
                    logger.warning(f"Semantic chunking failed: {e}. Falling back to traditional chunking")
                    text_splitter = get_text_splitter(use_semantic=False)
                    chunks = text_splitter.split_documents(documents)
                    logger.info(f"Created {len(chunks)} chunks using fallback traditional chunking")
                else:
                    raise e
            
            if not chunks:
                logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Get paths
            _, vector_db_path = get_file_paths(metadata.file_type, metadata.filename)
            
            # Ensure directory exists
            os.makedirs(vector_db_path, exist_ok=True)
            
            # Check if index exists
            index_exists = (
                os.path.exists(f"{vector_db_path}/index.faiss") and 
                os.path.exists(f"{vector_db_path}/index.pkl")
            )
            
            if index_exists:
                logger.info("Loading existing FAISS index")
                try:
                    db = FAISS.load_local(
                        vector_db_path, 
                        embedding_model,  # Dùng model đã có
                        allow_dangerous_deserialization=True
                    )
                    # Add new chunks to existing database
                    db.add_documents(chunks)
                    logger.info(f"Added {len(chunks)} chunks to existing database")
                except Exception as e:
                    logger.error(f"Failed to load existing index: {e}")
                    logger.info("Creating new FAISS index")
                    db = FAISS.from_documents(chunks, embedding_model)
            else:
                logger.info("Creating new FAISS index")
                db = FAISS.from_documents(chunks, embedding_model)
            
            # Save the database
            try:
                db.save_local(vector_db_path)
                logger.info(f"Successfully saved FAISS index to {vector_db_path}")
                
                # Verify files were created
                if os.path.exists(f"{vector_db_path}/index.faiss"):
                    faiss_size = os.path.getsize(f"{vector_db_path}/index.faiss")
                    logger.info(f"FAISS index file size: {faiss_size} bytes")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to save FAISS index: {e}")
                return False
            
    except Exception as e:
        logger.error(f"Error in add_to_embedding: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clear memory sau khi xong
        gc.collect()

def delete_from_faiss_index(vector_db_path: str, doc_id: str) -> bool:
    """Delete documents from FAISS index - optimized version"""
    try:
        index_path = f"{vector_db_path}/index.faiss"
        pkl_path = f"{vector_db_path}/index.pkl"
        
        if not (os.path.exists(index_path) and os.path.exists(pkl_path)):
            logger.warning(f"No FAISS index found at {vector_db_path}")
            return True
        
        # Use cached embedding model
        embedding_model = get_embedding_model()
        db = FAISS.load_local(
            vector_db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Find documents to delete
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        ids_to_delete = []
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                ids_to_delete.append(docstore_id)
        
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            db.save_local(vector_db_path)
            logger.info(f"Deleted {len(ids_to_delete)} documents with _id: {doc_id}")
        else:
            logger.warning(f"No documents found with _id: {doc_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting from FAISS index: {str(e)}")
        return False

def update_document_metadata_in_vector_store(doc_id: str, old_metadata: dict, new_metadata, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Update document by re-embedding - optimized version"""
    try:
        old_file_type = old_metadata.get('file_type')
        old_filename = old_metadata.get('filename')
        
        _, old_vector_db_path = get_file_paths(old_file_type, old_filename)
        
        # Delete old document
        success = delete_from_faiss_index(old_vector_db_path, doc_id)
        if not success:
            return False
        
        # Re-embed with new metadata
        file_path = new_metadata.url
        if os.path.exists(file_path):
            return add_to_embedding(file_path, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            logger.error(f"File not found for re-embedding: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata in vector store: {str(e)}")
        return False

def update_metadata_only(doc_id: str, new_metadata) -> bool:
    """Update only metadata without re-embedding - optimized version"""
    try:
        _, vector_db_path = get_file_paths(new_metadata.file_type, new_metadata.filename)
        
        if not (os.path.exists(f"{vector_db_path}/index.faiss") and 
                os.path.exists(f"{vector_db_path}/index.pkl")):
            logger.warning(f"Vector database not found at {vector_db_path}")
            return False
        
        # Use cached embedding model
        embedding_model = get_embedding_model()
        db = FAISS.load_local(
            vector_db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        
        # Update metadata
        docstore = db.docstore
        index_to_docstore_id = db.index_to_docstore_id
        updated_count = 0
        new_metadata_dict = new_metadata.dict(by_alias=True)
        
        for index, docstore_id in index_to_docstore_id.items():
            doc = docstore.search(docstore_id)
            if doc and doc.metadata.get('_id') == doc_id:
                doc.metadata.update(new_metadata_dict)
                docstore.add({docstore_id: doc})
                updated_count += 1
        
        if updated_count > 0:
            db.save_local(vector_db_path)
            logger.info(f"Updated metadata for {updated_count} chunks of document: {doc_id}")
            return True
        else:
            logger.warning(f"No chunks found for document: {doc_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating metadata only: {str(e)}")
        return False

def smart_metadata_update(doc_id: str, old_metadata: dict, new_metadata, force_re_embed: bool = False, use_semantic_chunking: bool = True, semantic_overlap: float = 0.2) -> bool:
    """Smart metadata update with fallback logic"""
    try:
        file_type_changed = old_metadata.get('file_type') != new_metadata.file_type
        filename_changed = old_metadata.get('filename') != new_metadata.filename
        
        if file_type_changed or filename_changed or force_re_embed:
            return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
        else:
            success = update_metadata_only(doc_id, new_metadata)
            if not success:
                logger.info("Metadata-only update failed, attempting re-embedding")
                return update_document_metadata_in_vector_store(doc_id, old_metadata, new_metadata, use_semantic_chunking, semantic_overlap)
            return success
            
    except Exception as e:
        logger.error(f"Error in smart metadata update: {str(e)}")
        return False

# Utility functions for performance monitoring
def get_embedding_model_info():
    """Get information about current embedding model"""
    global _embedding_model_cache
    return {
        "is_cached": _embedding_model_cache is not None,
        "cache_info": _create_embedding_model.cache_info() if hasattr(_create_embedding_model, 'cache_info') else None
    }

# Cleanup function for graceful shutdown
def cleanup_embedding_resources():
    """Cleanup embedding resources on shutdown"""
    logger.info("Cleaning up embedding resources...")
    clear_embedding_model_cache()