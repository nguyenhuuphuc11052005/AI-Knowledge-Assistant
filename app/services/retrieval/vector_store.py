from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from app.core.config import settings # File config hôm trước

def get_embedding_model():
    """
    Sử dụng model mã nguồn mở, chạy offline miễn phí 100%.
    BGE-small-en-v1.5 cực kỳ nhẹ và top đầu cho tiếng Anh.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'}, # Chạy bằng CPU của Mac là đủ nhanh
        encode_kwargs={'normalize_embeddings': True} # Giúp tính toán khoảng cách vector chính xác hơn
    )

def ingest_to_qdrant(chunks: List[Document], collection_name: str = "enterprise_knowledge"):
    """
    Lưu chunks vào Qdrant Database.
    """
    print(f"Đang tạo embeddings và lưu vào Qdrant tại {settings.QDRANT_URL}...")
    embeddings = get_embedding_model()
    
    # Langchain tự động lo việc gọi API Qdrant và đẩy dữ liệu lên
    qdrant = Qdrant.from_documents(
        chunks,
        embeddings,
        url=settings.QDRANT_URL,
        collection_name=collection_name,
        force_recreate=False # Đặt False để không ghi đè dữ liệu cũ nếu chạy nhiều lần
    )
    print("Lưu vào Vector DB thành công!")
    return qdrant