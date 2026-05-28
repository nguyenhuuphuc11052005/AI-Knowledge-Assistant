import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid

# Import các module từ backend của bạn
from app.core.config import settings
from app.services.workflow.rag_pipeline import KnowledgeAssistant
from app.models.domain import User, UserRole

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & KẾT NỐI DATABASE
# ==========================================
st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def init_db():
    """Tạo kết nối DB"""
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()

def ensure_test_users(db_session):
    """Đảm bảo cả 3 User mẫu (Intern, Manager, Admin) đều có sẵn trong DB"""
    test_users = [
        {"id": 1, "username": "test_intern", "role": UserRole.INTERN},
        {"id": 2, "username": "test_manager", "role": UserRole.MANAGER},
        {"id": 3, "username": "test_admin", "role": UserRole.ADMIN}
    ]
    
    for u in test_users:
        user_exists = db_session.query(User).filter(User.id == u["id"]).first()
        if not user_exists:
            new_user = User(id=u["id"], username=u["username"], role=u["role"])
            db_session.add(new_user)
    db_session.commit()

@st.cache_resource
def init_assistant():
    return KnowledgeAssistant()

db = init_db()
ensure_test_users(db) # Chạy hàm này ngay khi load app để sinh dữ liệu mồi
assistant = init_assistant()

# ==========================================
# 2. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# ==========================================
# Lưu lịch sử tin nhắn trên giao diện
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sinh ra một session_id ngẫu nhiên cho mỗi người mở web
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ==========================================
# 3. SIDEBAR: GIẢ LẬP ĐĂNG NHẬP (RBAC)
# ==========================================
with st.sidebar:
    st.title("⚙️ Bảng Điều Khiển")
    st.markdown("Giả lập quyền truy cập của người dùng để test tính năng RBAC.")
    
    # Cho phép chọn Role
    selected_role = st.selectbox(
        "Chọn vai trò của bạn:",
        ["intern", "manager", "admin"]
    )
    
    # Tự động đảm bảo User này tồn tại trong PostgreSQL
    user_id = {"intern": 1, "manager": 2, "admin": 3}[selected_role]
    
    # Nút xóa lịch sử chat
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown("**Hướng dẫn test:**\n- Dùng role **intern** hỏi về *Severance*.\n- Đổi sang **admin** hỏi lại câu đó.")

# ==========================================
# 4. GIAO DIỆN CHAT CHÍNH
# ==========================================
st.title("🏢 Enterprise AI Knowledge Assistant")
st.caption("Internal AI assistant. Responds based on authorized documents.")

# Render lại các tin nhắn cũ mỗi khi tải lại trang
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 5. XỬ LÝ LƯỢT CHAT MỚI
# ==========================================
# Khung nhập text cho user
if prompt := st.chat_input("Bạn muốn hỏi gì về quy định công ty?"):
    
    # Hiển thị câu hỏi của user lên màn hình
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Hiển thị biểu tượng AI đang suy nghĩ
    with st.chat_message("assistant"):
        with st.spinner("Đang lục tìm tài liệu và suy nghĩ..."):
            
            # Gọi RAG Pipeline (Xuyên thẳng xuống DB và LLM)
            try:
                answer = assistant.ask(
                    query=prompt, 
                    user_id=user_id, 
                    user_role=selected_role, 
                    session_id=st.session_state.session_id, 
                    db=db
                )
                
                # Render câu trả lời ra màn hình
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {str(e)}")