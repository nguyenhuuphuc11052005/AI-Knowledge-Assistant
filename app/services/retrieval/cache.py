import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import uuid

class QdrantSemanticCache:
    def __init__(self, embedding_model):
        self.client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        self.collection_name = "semantic_cache"
        self.embeddings = embedding_model
        self.threshold = 0.95 # Độ tương đồng ngữ nghĩa tối thiểu (95%)
        
        # Tự động tạo Collection cho Cache nếu chưa tồn tại
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=384, # Khớp với kích thước vector của BAAI/bge-small-en-v1.5
                    distance=qdrant_models.Distance.COSINE
                )
            )

    def get(self, query_text: str, user_role: str) -> str:
        """Tìm kiếm câu trả lời trong Cache dựa trên ngữ nghĩa câu hỏi và phân quyền"""
        # 1. Chuyển câu hỏi hiện tại thành Vector
        query_vector = self.embeddings.embed_query(query_text)
        
        # 2. Tìm kiếm trong collection cache kèm bộ lọc RBAC 
        # (Để tránh việc Intern ăn gian cache câu trả lời của Admin)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="allowed_role",
                        match=qdrant_models.MatchValue(value=user_role)
                    )
                ]
            ),
            limit=1
        )
        
        # 3. Nếu tìm thấy và độ tương đồng > 95%, trả về câu trả lời cũ ngay lập tức
        if search_result and search_result[0].score >= self.threshold:
            print(f"⚡ [CACHE HIT] Tìm thấy câu hỏi tương đồng ({search_result[0].score:.4f}). Đang xuất Cache...")
            return search_result[0].payload["cached_answer"]
            
        print("🔍 [CACHE MISS] Không có dữ liệu trong bộ đệm. Chuyển tiếp tới LLM...")
        return None

    def set(self, query_text: str, cached_answer: str, user_role: str):
        """Lưu cặp câu hỏi - câu trả lời mới vào bộ đệm ngữ nghĩa"""
        query_vector = self.embeddings.embed_query(query_text)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={
                        "original_question": query_text,
                        "cached_answer": cached_answer,
                        "allowed_role": user_role
                    }
                )
            ]
        )
        print("💾 [CACHE STORED] Đã lưu câu trả lời vào bộ đệm ngữ nghĩa.")