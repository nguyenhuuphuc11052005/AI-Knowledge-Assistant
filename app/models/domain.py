import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class UserRole(str, enum.Enum):
    INTERN = "intern"
    MANAGER = "manager"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.INTERN, nullable=False)
    
    # Quan hệ 1-N với bảng Conversation
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String(100), index=True, nullable=False) # ID phiên chat để track memory
    
    # Lưu câu hỏi của user và câu trả lời của LLM
    user_query = Column(Text, nullable=False)
    system_response = Column(Text, nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Khai báo quan hệ ngược lại
    user = relationship("User", back_populates="conversations")