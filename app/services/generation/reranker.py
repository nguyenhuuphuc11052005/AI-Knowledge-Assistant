from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class LocalReranker:
    def __init__(self):
        # Tải mô hình Reranker miễn phí từ HuggingFace về máy chạy local
        print("Đang khởi tạo Local Reranker (BAAI/bge-reranker-base)...")
        self.model = CrossEncoder("BAAI/bge-reranker-base", max_length=512, device="cpu")

    def rerank(self, query: str, documents: List[Any], top_n: str = 4) -> List[Any]:
        """
        Nhận vào câu hỏi và danh sách các chunk, trả về top N chunk có điểm số cao nhất.
        """
        if not documents:
            return []

        # Chuẩn bị cặp dữ liệu (Query, Text) cho mô hình Cross-Encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Mô hình chấm điểm chấm điểm tương quan sâu
        scores = self.model.predict(pairs)
        
        # Gắn điểm số vào metadata của từng document để tiện theo dõi
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            
        # Sắp xếp danh sách document theo thứ tự điểm giảm dần
        ranked_documents = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        
        # Trả về top N kết quả xuất sắc nhất
        return ranked_documents[:top_n]