from sqlalchemy.orm import Session
from app.models.domain import Conversation

def get_chat_history(db: Session, session_id: str, limit: int = 4) -> str:
    """
    Lấy 4 lượt chat gần nhất từ DB và format thành chuỗi văn bản.
    """
    # Lấy dữ liệu từ bảng Conversation, sắp xếp theo thời gian tăng dần
    past_chats = db.query(Conversation)\
                   .filter(Conversation.session_id == session_id)\
                   .order_by(Conversation.created_at.desc())\
                   .limit(limit)\
                   .all()
    
    # Do lấy desc() nên phải đảo ngược lại để đúng thứ tự thời gian
    past_chats.reverse()
    
    if not past_chats:
        return "Không có lịch sử trò chuyện."

    history_str = ""
    for chat in past_chats:
        history_str += f"Nhân viên: {chat.user_query}\n"
        history_str += f"AI: {chat.system_response}\n\n"
        
    return history_str