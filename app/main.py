from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.services.workflow.rag_pipeline import KnowledgeAssistant
from app.services.retrieval.cache import QdrantSemanticCache # Import Cache mới
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
from dotenv import load_dotenv

# Tìm đường dẫn tuyệt đối trỏ thẳng ra file .env ở thư mục gốc dự án
base_dir = Path(__file__).resolve().parent.parent
env_path = base_dir / ".env"
import os
# Ép nạp file .env một cách tường minh
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    print(f"⚠️ Cảnh báo: Không tìm thấy file .env tại {env_path}")



engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
SessionLocal = sessionmaker(bind=engine)

app = FastAPI(title="Enterprise RAG API v3 (With Semantic Cache)")
assistant = KnowledgeAssistant()

# Khởi tạo mô hình Embedding giống hệt trong RAG pipeline để đồng bộ không gian vector
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
semantic_cache = QdrantSemanticCache(embedding_model=embedding_model)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ChatRequest(BaseModel):
    query: str
    user_id: int
    user_role: str
    session_id: str

@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    
    # 🕵️ BƯỚC 1: KIỂM TRA SEMANTIC CACHE TRƯỚC
    cached_answer = semantic_cache.get(query_text=request.query, user_role=request.user_role)
    if cached_answer:
        # Nếu Cache Hit, trả về ngay lập tức (Thời gian xử lý ~ 0.01 giây)
        return {"answer": cached_answer, "session_id": request.session_id, "source": "Semantic Cache"}

    # 🤖 BƯỚC 2: CACHE MISS ➡️ CHẠY LUỒNG RAG TRUYỀN THỐNG
    try:
        # Gọi luồng RAG xử lý (Qdrant Search -> Reranker -> Llama 3.2)
        answer = assistant.ask(
            query=request.query,
            user_id=request.user_id,
            user_role=request.user_role,
            session_id=request.session_id,
            db=db
        )
        
        # 💾 BƯỚC 3: GHI LẠI KẾT QUẢ VÀO CACHE CHO LẦN HỎI SAU
        semantic_cache.set(
            query_text=request.query,
            cached_answer=answer,
            user_role=request.user_role
        )
        
        return {"answer": answer, "session_id": request.session_id, "source": "LLM Generation"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")