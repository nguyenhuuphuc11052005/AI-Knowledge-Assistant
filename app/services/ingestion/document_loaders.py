from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from typing import List

def load_pdf_document(file_path: str) -> List[Document]:
    """
    Hàm đọc file PDF và trả về danh sách các trang (Document).
    Mỗi Document chứa text và metadata mặc định (số trang, tên file).
    """
    print(f"Đang đọc file: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents