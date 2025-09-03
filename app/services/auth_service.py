# app/auth/jwt_auth.py
import time
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import logging
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)
security = HTTPBearer()

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "secret_key")
JWT_ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Xác thực JWT token và trả về payload
    """
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
       
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
def verify_token_v2(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Xác thực JWT token v2 cho cấu trúc payload mới và trả về payload
    Kiểm tra token validity, expiration, và cấu trúc payload
    """
    try:
        token = credentials.credentials
        
        # Decode JWT token
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        
        # Kiểm tra cấu trúc payload có đúng format mong muốn không
        required_fields = ["username", "department", "user_type", "full_name"]
        missing_fields = [field for field in required_fields if field not in payload]
        
        if missing_fields:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token structure. Missing fields: {missing_fields}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Kiểm tra các giá trị không được rỗng
        empty_fields = [field for field in required_fields if not payload.get(field)]
        if empty_fields:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token data. Empty fields: {empty_fields}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Kiểm tra thêm các field chuẩn JWT nếu có
        current_time = int(time.time())
        
        # Kiểm tra exp (expiration time)
        if "exp" in payload:
            if current_time >= payload["exp"]:
                raise HTTPException(
                    status_code=401,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Kiểm tra nbf (not before)
        if "nbf" in payload:
            if current_time < payload["nbf"]:
                raise HTTPException(
                    status_code=401,
                    detail="Token not yet valid",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Kiểm tra iat (issued at) - không quá xa trong quá khứ
        if "iat" in payload:
            # Token không được issue quá 24 giờ trước (tùy chỉnh theo nhu cầu)
            max_age = 24 * 60 * 60  # 24 hours
            if current_time - payload["iat"] > max_age:
                raise HTTPException(
                    status_code=401,
                    detail="Token is too old",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        # Kiểm tra token type nếu có
        if "type" in payload:
            if payload["type"] != "access":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        # Re-raise HTTPExceptions (đã được xử lý ở trên)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
def check_file_access_permission(user_payload: Dict[str, Any], file_metadata: Dict[str, Any]) -> bool:
    """
    Xác thực quyền truy cập cho JWT payload có cấu trúc mới
    JWT payload format:
    {
        "username": "teacher001",
        "user_type": "Giáo viên", 
        "full_name": "TS. Lê Văn Cường",
        "department": "CNTT"
    }
    
    Args:
        user_payload: Payload từ JWT token
        file_metadata: Metadata của file chứa thông tin quyền truy cập
    
    Returns:
        bool: True nếu có quyền truy cập, False nếu không
    """
    try:
        # Lấy thông tin user từ token (cấu trúc mới)
        user_id = str(user_payload.get("username", ""))
        user_department = user_payload.get("department", "")
        user_type = user_payload.get("user_type", "")
        full_name = user_payload.get("full_name", "")
        
        
        # Kiểm tra xem file có metadata role không
        role_info = file_metadata.get("role", {})
        
        # Nếu role rỗng hoặc không tồn tại -> cho phép truy cập
        if not role_info:
            return True
        
        # Lấy danh sách allowed users và subjects
        allowed_users = role_info.get("user", [])
        allowed_subjects = role_info.get("subject", [])
        
        # Nếu cả user và subject đều rỗng -> cho phép truy cập
        if not allowed_users and not allowed_subjects:
            return True
        
        # Kiểm tra logic phân quyền - CHỈ CẦN THỎA MÃN MỘT TRONG HAI
        user_check_passed = False
        department_check_passed = False
        
        # Kiểm tra user_id nếu có danh sách allowed_users
        if allowed_users:
            user_check_passed = user_id in allowed_users
        
        # Kiểm tra department nếu có danh sách allowed_subjects  
        if allowed_subjects:
            department_check_passed = user_department in allowed_subjects
        
        # Logic OR: Chỉ cần một trong hai điều kiện được thỏa mãn
        has_access = False
        
        # Nếu có cấu hình user và user được phép -> cho phép truy cập
        if allowed_users and user_check_passed:
            has_access = True
        
        # Nếu có cấu hình subject và department được phép -> cho phép truy cập  
        if allowed_subjects and department_check_passed:
            has_access = True
        
        if has_access:
            logger.info(f"Access GRANTED for user {user_id} ({full_name}) with department {user_department}")
        else:
            logger.warning(f"Access DENIED for user {user_id} ({full_name}) with department {user_department}")
            logger.warning(f"User check: {'PASSED' if user_check_passed else 'FAILED'}, Department check: {'PASSED' if department_check_passed else 'FAILED'}")
        
        return has_access
        
    except Exception as e:
        logger.error(f"Error checking file access permission v2: {str(e)}")
        return False
def verify_and_check_file_access(
    file_metadata: Dict[str, Any], 
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Kết hợp xác thực token và kiểm tra quyền truy cập file
    
    Args:
        file_metadata: Metadata của file
        credentials: JWT credentials
    
    Returns:
        Dict: User payload nếu xác thực và có quyền truy cập thành công
    
    Raises:
        HTTPException: Nếu không có quyền truy cập hoặc token không hợp lệ
    """
    # Xác thực token
    user_payload = verify_token(credentials)
    
    # Kiểm tra quyền truy cập file
    if not check_file_access_permission(user_payload, file_metadata):
        raise HTTPException(
            status_code=403,
            detail="Bạn không có quyền truy cập vào tài liệu này",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_payload


def verify_and_check_file_access(
    file_metadata: Dict[str, Any], 
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Kết hợp xác thực token v2 và kiểm tra quyền truy cập file cho JWT v2
    
    Args:
        file_metadata: Metadata của file
        credentials: JWT credentials
    
    Returns:
        Dict: User payload nếu xác thực và có quyền truy cập thành công
    
    Raises:
        HTTPException: Nếu không có quyền truy cập hoặc token không hợp lệ
    """
    # Xác thực token với hàm v2
    user_payload = verify_token_v2(credentials)
    
    # Kiểm tra quyền truy cập file
    if not check_file_access_permission(user_payload, file_metadata):
        raise HTTPException(
            status_code=403,
            detail="Access denied. You don't have permission to access this document.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user_payload

def filter_accessible_files(
    user_payload: Dict[str, Any], 
    search_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Lọc danh sách kết quả search để chỉ trả về các file mà user có quyền truy cập
    
    Args:
        user_payload: Payload từ JWT token
        search_results: Danh sách kết quả search với metadata
    
    Returns:
        List: Danh sách các file được phép truy cập
    """
    accessible_files = []
    
    for result in search_results:
        # Giả sử mỗi result có trường metadata chứa thông tin file
        file_metadata = result.get("metadata", {})
        
        if check_file_access_permission(user_payload, file_metadata):
            accessible_files.append(result)
        else:
            logger.info(f"Filtered out file: {file_metadata.get('filename', 'unknown')}")
    
    return accessible_files

