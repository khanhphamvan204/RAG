# app/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    DATA_PATH = os.getenv("DATA_PATH", "data")
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/db_faiss")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017/")
    
    FILE_TYPE_PATHS = {
        'public': {
            'file_folder': os.getenv("PUBLIC_FILE_FOLDER", "Public_Rag_Info/File_Folder"),
            'vector_folder': os.getenv("PUBLIC_VECTOR_FOLDER", "Public_Rag_Info/Faiss_Folder")
        },
        'student': {
            'file_folder': os.getenv("STUDENT_FILE_FOLDER", "Student_Rag_Info/File_Folder"),
            'vector_folder': os.getenv("STUDENT_VECTOR_FOLDER", "Student_Rag_Info/Faiss_Folder")
        },
        'teacher': {
            'file_folder': os.getenv("TEACHER_FILE_FOLDER", "Teacher_Rag_Info/File_Folder"),
            'vector_folder': os.getenv("TEACHER_VECTOR_FOLDER", "Teacher_Rag_Info/Faiss_Folder")
        },
        'admin': {
            'file_folder': os.getenv("ADMIN_FILE_FOLDER", "Admin_Rag_Info/File_Folder"),
            'vector_folder': os.getenv("ADMIN_VECTOR_FOLDER", "Admin_Rag_Info/Faiss_Folder")
        }
    }