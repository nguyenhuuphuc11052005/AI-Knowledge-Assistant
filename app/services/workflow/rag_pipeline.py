from sqlalchemy.orm import Session
from langchain_community.chat_models import ChatOllama
from app.services.retrieval.search import advanced_search
from app.services.workflow.memory import get_chat_history
from app.services.generation.prompts import rag_prompt
from app.models.domain import Conversation

class KnowledgeAssistant:
    def __init__(self):
        # Khởi tạo mô hình llama3.2 chạy local thông qua Ollama
        self.llm = ChatOllama(model="llama3.2", temperature=0.1) 
        # temperature=0.1 giúp câu trả lời nhất quán, bớt sáng tạo/ảo giác

    def ask(self, query: str, user_id: int, user_role: str, session_id: str, db: Session):
        print(f"\n--- Bắt đầu xử lý luồng cho User: {user_id} | Role: {user_role} ---")
        
        # 1. Kéo lịch sử chat từ PostgreSQL
        history = get_chat_history(db, session_id)
        
        # 2. Tìm kiếm tài liệu từ Qdrant
        retrieved_docs = advanced_search(query=query, user_role=user_role)
        
        # Format tài liệu thành chuỗi text kèm Trích dẫn (Citation)
        # Sử dụng thẻ <doc> để LLM dễ nhận diện ranh giới tài liệu
        context_str = ""
        for i, doc in enumerate(retrieved_docs):
            doc_id = i + 1
            source_file = doc.metadata.get("source_file", "Unknown Source")
            
            context_str += f"<doc id=\"{doc_id}\">\n"
            context_str += f"Source: {source_file}\n"
            context_str += f"Content: {doc.page_content}\n"
            context_str += "</doc>\n\n"
            
        if not context_str:
            context_str = "No relevant documents found."

        # 3. Ép vào Prompt Template
        final_prompt = rag_prompt.format(
            context=context_str,
            history=history,
            query=query
        )
        
        # 4. Gửi cho LLM sinh câu trả lời
        print("Đang gọi LLM (Ollama) sinh câu trả lời...")
        response = self.llm.invoke(final_prompt)
        answer = response.content
        
        # 5. Lưu lại đoạn chat này vào PostgreSQL để làm Memory cho lượt hỏi sau
        new_chat = Conversation(
            user_id=user_id,
            session_id=session_id,
            user_query=query,
            system_response=answer
        )
        db.add(new_chat)
        db.commit()
        
        return answer