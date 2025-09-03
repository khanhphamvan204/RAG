# app/services/file_service.py
import os
from fastapi import HTTPException
from app.config import Config
import logging

logger = logging.getLogger(__name__)

def get_file_paths(file_type: str, filename: str) -> tuple[str, str]:
    if file_type not in Config.FILE_TYPE_PATHS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file_type: {file_type}. Must be one of: {list(Config.FILE_TYPE_PATHS.keys())}"
        )
    
    base_path = Config.DATA_PATH
    file_folder = Config.FILE_TYPE_PATHS[file_type]['file_folder']
    vector_folder = Config.FILE_TYPE_PATHS[file_type]['vector_folder']
    
    file_path = os.path.join(base_path, file_folder, filename).replace("\\", "/")
    vector_db_path = os.path.join(base_path, vector_folder).replace("\\", "/")
    
    return file_path, vector_db_path