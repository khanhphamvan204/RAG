# app/routes/documents.py
from fastapi import APIRouter, HTTPException, Depends
import logging
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from app.config import Config
import os
import json
from app.services.auth_service import verify_token

router = APIRouter()
logger = logging.getLogger(__name__)

# Sử dụng connection pooling
_mongo_client = None

def get_mongo_client():
    """Get MongoDB client with connection pooling"""
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                Config.DATABASE_URL,
                maxPoolSize=10,
                minPoolSize=2,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000
            )
            # Test connection
            _mongo_client.admin.command('ping')
            logger.info("MongoDB connection established in documents route")
        except Exception as e:
            logger.error(f"Failed to establish MongoDB connection: {e}")
            _mongo_client = None
            raise
    return _mongo_client

@router.get("/types", response_model=dict)
async def get_file_types(current_user: dict = Depends(verify_token)):
    """Lấy danh sách các loại file được hỗ trợ."""
    return {
        "file_types": [
            {"value": "public", "label": "Thông báo chung (Public)", "description": "Tài liệu công khai cho tất cả người dùng"},
            {"value": "student", "label": "Sinh viên (Student)", "description": "Tài liệu dành cho sinh viên"},
            {"value": "teacher", "label": "Giảng viên (Teacher)", "description": "Tài liệu dành cho giảng viên"},
            {"value": "admin", "label": "Quản trị viên (Admin)", "description": "Tài liệu dành cho quản trị viên"}
        ]
    }

@router.get("/list", response_model=dict)
async def list_documents(current_user: dict = Depends(verify_token), file_type: str = None, limit: int = 100, skip: int = 0):
    """Lấy danh sách tài liệu."""
    try:
        documents = []
        
        # Thử lấy từ MongoDB trước
        try:
            client = get_mongo_client()  # Sử dụng pooled connection
            db = client["faiss_db"]
            collection = db["metadata"]
            
            filter_dict = {"file_type": file_type} if file_type else {}
            
            documents = list(collection.find(filter_dict).skip(skip).limit(limit).sort("createdAt", -1))
            
            
            if documents:
                return {"documents": documents, "total": len(documents), "source": "mongodb"}
                
        except PyMongoError as e:
            logger.error(f"Failed to retrieve documents from MongoDB: {str(e)}")
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
        
        # Fallback: Lấy từ metadata.json
        logger.info("Falling back to JSON files")
        base_path = Config.DATA_PATH
        metadata_paths = [
            f"{base_path}/{Config.FILE_TYPE_PATHS[role]['vector_folder']}/metadata.json"
            for role in Config.FILE_TYPE_PATHS
        ]
        
        for metadata_file in metadata_paths:
            try:
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_list = json.load(f)
                    
                    if file_type:
                        metadata_list = [item for item in metadata_list if item.get('file_type') == file_type]
                    
                    documents.extend(metadata_list)
            except Exception as e:
                logger.error(f"Error reading {metadata_file}: {str(e)}")
        
        # Sort and paginate
        documents.sort(key=lambda x: x.get('createdAt', ''), reverse=True)
        total = len(documents)
        documents = documents[skip:skip + limit]
        
        return {"documents": documents, "total": total, "source": "json", "showing": len(documents)}
    
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

# Optional: Add cleanup function for app shutdown
def close_documents_mongo():
    """Close MongoDB connection for documents route"""
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None