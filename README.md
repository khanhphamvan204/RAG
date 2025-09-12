# 🤖 RAG - Hệ thống Quản lý Tài liệu và Vector Embedding Thông minh

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-brightgreen.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**RAG (Retrieval-Augmented Generation) API Management System** là một ứng dụng web hiện đại được xây dựng bằng FastAPI, cho phép quản lý tài liệu và tạo vector embeddings một cách hiệu quả với sự hỗ trợ của FAISS vector search engine.

## ✨ Tính năng chính

- 📤 **Upload đa định dạng**: Hỗ trợ PDF, TXT, DOCX, CSV, XLSX, XLS
- 🤖 **AI-powered OCR**: Sử dụng PaddleOCR cho nhận dạng văn bản chính xác
- 📊 **Xử lý bảng thông minh**: Tự động trích xuất và xử lý bảng từ PDF
- 🔍 **Vector Search**: Tìm kiếm ngữ nghĩa (semantic search) với FAISS
- 👥 **Hệ thống phân quyền**: Quản lý quyền truy cập chi tiết theo user và subject
- 🚀 **API RESTful**: Tích hợp dễ dàng với các hệ thống khác

## 🏗️ Kiến trúc hệ thống

```
📁 Root_Folder/
├── 🌐 Public_Rag_Info/     # Tài liệu công khai
│   ├── File_Folder/        # Lưu trữ files gốc
│   └── Faiss_Folder/       # Vector database
│       ├── index.faiss
│       ├── index.pkl
│       └── metadata.json
├── 🎓 Student_Rag_Info/    # Tài liệu sinh viên
├── 👨🏫 Teacher_Rag_Info/   # Tài liệu giảng viên  
└── ⚙️ Admin_Rag_Info/      # Tài liệu quản trị
```

## 🔧 Yêu cầu hệ thống

- **Python**: 3.8+
- **MongoDB**: 4.4+
- **RAM**: 4GB (khuyến nghị)
- **Dung lượng**: 4GB trống
- **OS**: Windows, macOS, Linux

## ⚡ Cài đặt nhanh

### 1. Clone repository

```bash
git clone https://github.com/khanhphamvan204/RAG.git
cd RAG
```

### 2. Tạo và kích hoạt virtual environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

Tạo file `.env` trong thư mục gốc:

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Model Paths
MODEL_EMBEDDING=model/vinallama-7b-chat_q5_0.gguf
MODEL_PADDLEOCR=model/.paddlex

# Database & Storage
DATABASE_URL=mongodb://localhost:27017/
DATA_PATH=Root_Folder
VECTOR_DB_PATH=vectorstore
```

### 5. Khởi động MongoDB

```bash
# Ubuntu/Debian
sudo systemctl start mongod

# macOS với Homebrew
brew services start mongodb-community

# Windows
net start MongoDB
```

### 6. Chạy ứng dụng

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
python app/main.py
```

API sẽ được khởi chạy tại: `http://localhost:8000`

## 📋 API Endpoints

### Health Check
```bash
GET /health
```

### Quản lý tài liệu

#### Upload tài liệu
```bash
POST /documents/vector/add
Content-Type: multipart/form-data

Parameters:
- file: File upload (required)
- uploaded_by: string (required)
- file_type: "public"|"student"|"teacher"|"admin" (required)
- role_user: JSON string array (optional)
- role_subject: JSON string array (optional)
```

#### Lấy danh sách tài liệu
```bash
GET /documents/list?file_type={type}&limit={limit}&skip={skip}

Parameters:
- file_type: string (optional) - Lọc theo loại file
- limit: integer (optional, default: 100) - Số lượng trả về
- skip: integer (optional, default: 0) - Bỏ qua số lượng
```

#### Xóa tài liệu
```bash
DELETE /documents/vector/{doc_id}

Parameters:
- doc_id: string (required) - ID của document
```

#### Lấy danh sách loại file hỗ trợ
```bash
GET /documents/types
```

## 🔧 Sử dụng API

### Python Example

```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
data = {
    'uploaded_by': 'Test User',
    'file_type': 'public',
    'role_user': '["user_001", "user_002"]',
    'role_subject': '["cntt", "toan"]'
}

response = requests.post('http://localhost:8000/documents/vector/add',
                        files=files, data=data)
print(response.json())

# List documents
response = requests.get('http://localhost:8000/documents/list?file_type=public&limit=10')
documents = response.json()['documents']
print(documents)
```

### cURL Examples

```bash
# Upload file
curl -X POST "http://localhost:8000/documents/vector/add" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "uploaded_by=Test User" \
  -F "file_type=public" \
  -F "role_user=[\"user_001\"]"

# List documents
curl -X GET "http://localhost:8000/documents/list?file_type=public&limit=5"

# Delete document
curl -X DELETE "http://localhost:8000/documents/vector/{doc_id}"
```

## ⚙️ Biến môi trường

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `GEMINI_API_KEY` | Google Generative AI API key | Required |
| `DATA_PATH` | Thư mục lưu trữ data | Root_Folder |
| `VECTOR_DB_PATH` | Thư mục vector database | vectorstore |
| `MODEL_EMBEDDING` | Đường dẫn model embedding | model/vinallama-7b-chat_q5_0.gguf |
| `MODEL_PADDLEOCR` | Đường dẫn model PaddleOCR | model/.paddlex |
| `DATABASE_URL` | MongoDB connection string | mongodb://localhost:27017/ |

## 🐳 Docker Deployment

## 🐳 Docker Setup

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Cài các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và cài python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Copy source code
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```

### docker-compose.yml

```yaml
version: '3.8'

services:
  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rag-api-main
    ports:
      - "8000:8000"
    volumes:
      - ./Root_Folder:/app/Root_Folder
      - ./model:/app/model
      - ./logs:/app/logs
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
      - DATABASE_URL=mongodb://admin:123@mongo:27017/faiss_db?authSource=admin
    networks:
      - app-network
    depends_on:
      mongo:
        condition: service_healthy
    restart: unless-stopped

  # MongoDB Database
  mongo:
    image: mongo:6.0
    container_name: rag-mongo-db
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
      - ./mongo-init:/docker-entrypoint-initdb.d
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=123
      - MONGO_INITDB_DATABASE=faiss_db
    healthcheck:
      test: |
        mongosh --host localhost \
                --port 27017 \
                --username admin \
                --password 123 \
                --authenticationDatabase admin \
                --eval "db.adminCommand('ping')"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 20s
    networks:
      - app-network
    restart: unless-stopped

volumes:
  mongo-data:
    driver: local

networks:
  app-network:
    driver: bridge
```

### Cách sử dụng

#### 1. Chuẩn bị files

Đảm bảo bạn có cấu trúc thư mục như sau:
```
RAG/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── app/
│   └── main.py
├── Root_Folder/
├── model/
```

#### 2. Cấu hình .env

```env
# API Keys
GEMINI_API_KEY=your_gemini_api_key_here

# Database (cho Docker)
DATABASE_URL=mongodb://admin:123@mongo:27017/faiss_db?authSource=admin

# Model Paths
MODEL_EMBEDDING=model/vinallama-7b-chat_q5_0.gguf
MODEL_PADDLEOCR=model/.paddlex

# Storage Paths
DATA_PATH=Root_Folder
VECTOR_DB_PATH=vectorstore
```

#### 3. Chạy với Docker Compose

```bash
# Build và khởi động tất cả services
docker-compose up --build

# Chạy ở background
docker-compose up -d --build

# Xem logs
docker-compose logs -f

# Xem logs của service cụ thể
docker-compose logs -f app
docker-compose logs -f mongo

# Dừng services
docker-compose down

# Dừng và xóa volumes (cẩn thận - sẽ mất data)
docker-compose down -v
```

#### 4. Kiểm tra services

```bash
# Kiểm tra container đang chạy
docker ps

# Kiểm tra health của MongoDB
docker-compose exec mongo mongosh --username admin --password 123 --authenticationDatabase admin

# Test API endpoint
curl http://localhost:8000/health

# Vào container app để debug
docker-compose exec app bash
```

#### 5. Development với Docker

```bash
# Chỉ rebuild app service
docker-compose up --build app

# Xem logs real-time
docker-compose logs -f app

# Restart service
docker-compose restart app
```

### Troubleshooting

#### MongoDB connection issues
```bash
# Kiểm tra MongoDB logs
docker-compose logs mongo

# Test kết nối MongoDB
docker-compose exec app python -c "
from pymongo import MongoClient
try:
    client = MongoClient('mongodb://admin:123@mongo:27017/faiss_db?authSource=admin')
    print('Connected:', client.server_info())
except Exception as e:
    print('Error:', e)
"
```

#### Volume permissions
```bash
# Fix permissions cho Root_Folder
sudo chown -R $USER:$USER Root_Folder model logs

# Hoặc sử dụng docker user
docker-compose exec app chown -R root:root /app/Root_Folder
```

#### Memory issues
```yaml
# Thêm vào docker-compose.yml trong service app:
deploy:
  resources:
    limits:
      memory: 4G
    reservations:
      memory: 2G
```

### Server deployment

```yaml
# docker-compose.remote.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: faiss-api-main
    ports:
      - "8000:8000"
    volumes:
      - ./Root_Folder:/app/Root_Folder
      - ./.env:/app/.env
    environment:
      - PYTHONUNBUFFERED=1
      - DATABASE_URL=mongodb://admin:123@ai-database.bitech.vn:27017/faiss_db?authSource=admin
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

```

```bash
# Deploy production
docker-compose -f docker-compose.remote.yml up -d --build
```

## 📊 Performance

- **Upload speed**: ~2-5MB/s tùy file type
- **OCR processing**: ~1-3 pages/second
- **Vector search**: <100ms cho 10K documents
- **Memory usage**: ~1-2GB với 1000 documents

## ⚡ Tối ưu hóa

- Sử dụng GPU cho PaddleOCR nếu có
- Tăng `chunk_size` cho file lớn
- Enable MongoDB indexing
- Sử dụng Redis cache cho metadata

## 🔍 Troubleshooting

### MongoDB connection issues
```bash
# Kiểm tra MongoDB đang chạy
sudo systemctl status mongod

# Khởi động MongoDB
sudo systemctl start mongod
```

### Memory issues
- Tăng RAM cho hệ thống
- Giảm `chunk_size` trong embedding.py
- Xử lý file theo batch nhỏ hơn

### FAISS index corruption
```bash
# Xóa và tạo lại index
rm -rf Root_Folder/*/Faiss_Folder/index.*
# Upload lại documents
```

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Contributing

1. Fork project
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [LangChain](https://langchain.com/) - LLM framework  
- [FAISS](https://faiss.ai/) - Vector similarity search
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [MongoDB](https://mongodb.com/) - Database