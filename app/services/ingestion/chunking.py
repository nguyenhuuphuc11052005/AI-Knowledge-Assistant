from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def split_documents(documents: List[Document], allowed_roles: List[str]) -> List[Document]:
    """
    Cắt nhỏ văn bản thành các chunk và tiêm thêm metadata RBAC.
    """
    # chunk_size: Kích thước tối đa của 1 chunk (ký tự)
    # chunk_overlap: Số ký tự gối lên nhau giữa 2 chunk liên tiếp
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "] # Ưu tiên cắt ở đoạn văn, rồi đến câu, rồi đến từ
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Tiêm thêm custom metadata (quan trọng cho phân quyền)
    for chunk in chunks:
        chunk.metadata["allowed_roles"] = allowed_roles
        
    print(f"Đã chia thành {len(chunks)} chunks.")
    return chunks