from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from langchain_qdrant import QdrantVectorStore 
from app.core.config import settings
from app.services.retrieval.vector_store import get_embedding_model
from app.services.generation.reranker import LocalReranker
from typing import List, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# Khởi tạo Reranker dùng chung
reranker_client = LocalReranker()

def advanced_search(query: str, user_role: str, collection_name: str = "enterprise_knowledge") -> List[Any]:
    """
    Hàm thực hiện Hybrid Search có kèm bộ lọc phân quyền RBAC và Reranking.
    """
    # 1. Kết nối tới Qdrant Client bằng thư viện gốc
    client = QdrantClient(url=settings.QDRANT_URL)
    embeddings = get_embedding_model()
    
    # 2. Xây dựng bộ lọc phân quyền (RBAC Filter)
    rbac_filter = qdrant_models.Filter(
        must=[
            qdrant_models.FieldCondition(
                key="metadata.allowed_roles",
                match=qdrant_models.MatchValue(value=user_role)
            )
        ]
    )
    
    # 3. Thực hiện truy xuất từ Vector Store bằng QdrantVectorStore (Cập nhật API mới)
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=settings.QDRANT_URL
    )
    
    print(f"[{user_role}] Đang tìm kiếm vector kết hợp lọc RBAC cho câu hỏi: '{query}'...")
    
    # Tìm kiếm và áp filter
    raw_results = vector_store.similarity_search(
        query=query,
        k=20, 
        filter=rbac_filter 
    )
    
    # 4. Đưa top 20 qua bộ lọc Reranker để lấy ra 4 kết quả tinh túy nhất
    print(f"Đang tiến hành Rerank trên {len(raw_results)} tài liệu thu thập được...")
    final_context = reranker_client.rerank(query=query, documents=raw_results, top_n=4)
    
    return final_context