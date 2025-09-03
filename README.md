# 🚀 FAISS API Management System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)](https://mongodb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Hệ thống quản lý tài liệu và vector embedding thông minh với FAISS**

[Tính năng](#-tính-năng) • [Cài đặt](#-cài-đặt) • [Sử dụng](#-sử-dụng) • [API](#-api-endpoints) • [Cấu hình](#-cấu-hình)

</div>

---

## 📋 Tổng quan

FAISS API Management System là một ứng dụng web hiện đại được xây dựng bằng FastAPI, cho phép quản lý tài liệu và tạo vector embeddings một cách hiệu quả. Hệ thống hỗ trợ nhiều định dạng file và cung cấp giao diện web trực quan để quản lý.

### 🎯 Mục tiêu
- **Quản lý tài liệu**: Upload, lưu trữ và tổ chức tài liệu theo loại
- **Vector Embedding**: Tạo và lưu trữ embeddings sử dụng Google Generative AI
- **Phân quyền**: Quản lý quyền truy cập dựa trên user và subject
- **Tìm kiếm**: Tìm kiếm tài liệu thông minh với FAISS vector store

## ✨ Tính năng

### 🔧 Chức năng chính
- **📤 Upload đa định dạng**: PDF, TXT, DOCX, CSV, XLSX, XLS
- **🤖 AI-powered OCR**: Sử dụng PaddleOCR cho nhận dạng văn bản
- **📊 Xử lý bảng**: Tự động trích xuất và xử lý bảng từ PDF
- **🔍 Vector Search**: Tìm kiếm semantic với FAISS
- **👥 Phân quyền**: Quản lý quyền truy cập chi tiết
- **📱 Responsive UI**: Giao diện web hiện đại, thân thiện

### 🏗️ Kiến trúc hệ thống
```
📁 Root_Folder/
├── 🌐 Public_Rag_Info/     # Tài liệu công khai
├── 🎓 Student_Rag_Info/    # Tài liệu sinh viên
├── 👨‍🏫 Teacher_Rag_Info/    # Tài liệu giảng viên
└── ⚙️ Admin_Rag_Info/      # Tài liệu quản trị
```

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- MongoDB 4.4+
- 4GB RAM (khuyến nghị)
- 2GB dung lượng trống

### 1. Clone repository
```bash
git clone https://github.com/your-username/faiss-api.git
cd faiss-api
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường
Tạo file `.env` trong thư mục gốc:
```env
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_EMBEDDING=model/vinallama-7b-chat_q5_0.gguf
MODEL_PADDLEOCR=model/.paddlex
DATA_PATH=Root_Folder
VECTOR_DB_PATH=vectorstore
DATABASE_URL=mongodb://localhost:27017/
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

## 🎮 Sử dụng

### 🌐 Giao diện Web
Truy cập `http://localhost:8000` để sử dụng giao diện web:

#### 📤 Tab Upload Documents
1. **Chọn file**: Kéo thả hoặc click để chọn file
2. **Điền thông tin**: Người upload, loại file (Public/Student/Teacher/Admin)
3. **Phân quyền**: Chọn users và subjects có quyền truy cập
4. **Upload**: Click "Upload & Process" để xử lý

#### 📋 Tab Manage Documents
- **Xem danh sách**: Tất cả tài liệu với thống kê
- **Tìm kiếm**: Tìm theo tên file, người upload, ID
- **Lọc**: Lọc theo loại file
- **Quản lý**: Xem chi tiết và xóa tài liệu

### 🔧 API sử dụng

#### Upload tài liệu
```python
import requests

files = {'file': open('document.pdf', 'rb')}
data = {
    'uploaded_by': 'Nguyễn Văn A',
    'file_type': 'public',
    'role_user': '["user_001", "user_002"]',
    'role_subject': '["cntt", "toan"]'
}

response = requests.post('http://localhost:8000/documents/vector/add', 
                        files=files, data=data)
print(response.json())
```

#### Lấy danh sách tài liệu
```python
response = requests.get('http://localhost:8000/documents/list?file_type=public&limit=10')
documents = response.json()['documents']
```

## 📚 API Endpoints

### 🔍 Health Check
```http
GET /health
```
Kiểm tra trạng thái API.

### 📤 Upload Document
```http
POST /documents/vector/add
Content-Type: multipart/form-data

Parameters:
- file: File upload (required)
- uploaded_by: string (required)
- file_type: "public"|"student"|"teacher"|"admin" (required)
- role_user: JSON string array (optional)
- role_subject: JSON string array (optional)
```

### 📋 List Documents
```http
GET /documents/list?file_type={type}&limit={limit}&skip={skip}

Parameters:
- file_type: string (optional) - Lọc theo loại file
- limit: integer (optional, default: 100) - Số lượng trả về
- skip: integer (optional, default: 0) - Bỏ qua số lượng
```

### 🗑️ Delete Document
```http
DELETE /documents/vector/{doc_id}

Parameters:
- doc_id: string (required) - ID của document
```

### 📂 Get File Types
```http
GET /documents/types
```
Lấy danh sách các loại file được hỗ trợ.

## ⚙️ Cấu hình

### 🔐 Environment Variables

| Biến | Mô tả | Mặc định |
|------|-------|----------|
| `GEMINI_API_KEY` | Google Generative AI API key | **Required** |
| `DATA_PATH` | Thư mục lưu trữ data | `Root_Folder` |
| `VECTOR_DB_PATH` | Thư mục vector database | `vectorstore` |
| `MODEL_EMBEDDING` | Đường dẫn model embedding | `model/vinallama-7b-chat_q5_0.gguf` |
| `MODEL_PADDLEOCR` | Đường dẫn model PaddleOCR | `model/.paddlex` |
| `DATABASE_URL` | MongoDB connection string | `mongodb://localhost:27017/` |

### 📁 Cấu trúc thư mục
```
Root_Folder/
├── Public_Rag_Info/
│   ├── File_Folder/        # Files công khai
│   └── Faiss_Folder/       # Vector DB công khai
│       ├── index.faiss
│       ├── index.pkl
│       └── metadata.json
├── Student_Rag_Info/       # Tương tự cho Student
├── Teacher_Rag_Info/       # Tương tự cho Teacher
└── Admin_Rag_Info/         # Tương tự cho Admin
```

## 🛠️ Phát triển

### 🧪 Testing
```bash
# Chạy tests
pytest tests/

# Test với coverage
pytest --cov=app tests/
```

### 🐳 Docker
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build và chạy
docker build -t faiss-api .
docker run -p 8000:8000 -v $(pwd)/Root_Folder:/app/Root_Folder faiss-api
```

### 🔧 Logs
```bash
# Xem logs realtime
tail -f logs/app.log

# Cấu hình log level trong main.py
logging.basicConfig(level=logging.INFO)
```

## 🚨 Troubleshooting

### ❗ Lỗi thường gặp

#### MongoDB Connection Error
```bash
# Kiểm tra MongoDB đang chạy
sudo systemctl status mongod

# Khởi động MongoDB
sudo systemctl start mongod
```

#### PaddleOCR Installation Error
```bash
# Cài đặt dependencies cho PaddleOCR
pip install paddlepaddle-gpu  # Nếu có GPU
# hoặc
pip install paddlepaddle      # CPU only
```

#### Memory Error khi xử lý file lớn
- Tăng RAM cho hệ thống
- Giảm `chunk_size` trong `embedding.py`
- Xử lý file theo batch nhỏ hơn

#### FAISS Index Error
```bash
# Xóa và tạo lại index
rm -rf Root_Folder/*/Faiss_Folder/index.*
# Upload lại documents
```

## 📈 Performance

### 🎯 Metrics
- **Upload speed**: ~2-5MB/s tùy file type
- **OCR processing**: ~1-3 pages/second
- **Vector search**: <100ms cho 10K documents
- **Memory usage**: ~1-2GB với 1000 documents

### 🚀 Optimization
- Sử dụng GPU cho PaddleOCR nếu có
- Tăng `chunk_size` cho file lớn
- Enable MongoDB indexing
- Sử dụng Redis cache cho metadata

## 🤝 Đóng góp

1. Fork project
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [LangChain](https://langchain.com/) - LLM framework
- [FAISS](https://faiss.ai/) - Vector similarity search
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [MongoDB](https://mongodb.com/) - Database

---
