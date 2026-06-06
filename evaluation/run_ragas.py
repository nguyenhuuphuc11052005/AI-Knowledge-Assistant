import os
import sys

from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
# Tự động tìm đường dẫn thư mục gốc và add vào sys.path để Python nhận diện được folder 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
logging.basicConfig(level=logging.INFO)
import json
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv() 

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings # Dùng lại cục local xịn từ Giai đoạn 2
from ragas.run_config import RunConfig 
# Import RAGAS
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import hệ thống Backend của bạn
from app.services.workflow.rag_pipeline import KnowledgeAssistant
from app.services.retrieval.search import advanced_search
from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def run_evaluation():
     # 2. CẤU HÌNH BAN GIÁM KHẢO SỬ DỤNG MIX-MATCH
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ Lỗi: Bạn chưa cấu hình GEMINI_API_KEY trong file .env")
        return
    print("🚀 BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ VỚI CONTEXT THẬT (GEMINI + HF LOCAL)...")
    
    # [Giữ nguyên toàn bộ logic chạy vòng lặp thu thập câu trả lời bên dưới...]
    engine = create_engine(settings.SQLALCHEMY_DATABASE_URI)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    assistant = KnowledgeAssistant()
    
    with open("evaluation/ground_truth.json", "r") as f:
        test_data = json.load(f)
        
    data_samples = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [], 
        "reference": []
    }

    print("🤖 Đang chạy RAG Pipeline để thu thập dữ liệu thật...")
    for item in test_data:
        question = item["question"]
        print(f" -> Đang xử lý câu hỏi: {question}")
        
        answer = assistant.ask(query=question, user_id=3, user_role="admin", session_id="eval_123", db=db)
        actual_chunks = advanced_search(query=question, user_role="admin")
        context_texts = [doc.page_content for doc in actual_chunks]
        
        data_samples["user_input"].append(question)
        data_samples["response"].append(answer)
        data_samples["reference"].append(item["ground_truth"])
        data_samples["retrieved_contexts"].append(context_texts)
    print("🔍 KIỂM TRA DỮ LIỆU MẪU CỦA CÂU HỎI ĐẦU TIÊN:")
    print(f"Câu hỏi: {data_samples['user_input'][0]}")
    print(f"Bot trả lời: {data_samples['response'][0]}")
    print(f"Tài liệu Qdrant tìm được: {data_samples['retrieved_contexts'][0]}")
    print("-" * 50)
    dataset = Dataset.from_dict(data_samples)

   

    # print("⚖️ Đang kết nối tới Ban Giám Khảo (LLM: Llama 3.2 Local | Embeddings: HF BGE Local)...")

    # evaluator_llm = ChatOllama(
    #     model="llama3.2",
    #     temperature=0
    # )

    print("⚖️ Đang kết nối tới Ban Giám Khảo (LLM: Gemini 2.5 Flash | Embeddings: HF BGE Local)...")

    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=gemini_key,
        temperature=0
    )
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5") # Khởi chạy local cực mượt
    
    ragas_llm = LangchainLLMWrapper(evaluator_llm)
    ragas_emb = LangchainEmbeddingsWrapper(evaluator_embeddings)
    local_run_config = RunConfig(
        max_workers=1, 
        timeout=60, 
        max_retries=3
    )
    # 3. Bắt đầu chấm điểm
    print("📊 Giám khảo đang chấm điểm...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy
        ],
        llm=ragas_llm,
        embeddings=ragas_emb
    )
    
    # 4. Xuất báo cáo công khai
    print("\n✅ HOÀN TẤT ĐÁNH GIÁ THÀNH CÔNG!")
    df = result.to_pandas()
    df.to_csv("evaluation/rag_evaluation_report.csv", index=False)
    
    print("\n📊 BẢNG ĐIỂM HỆ THỐNG CỦA BẠN:")
    print(df[["user_input", "faithfulness", "answer_relevancy"]])

if __name__ == "__main__":
    run_evaluation()
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     google_api_key=os.getenv("GEMINI_API_KEY"),
    # )

    # response = llm.invoke("Hello")

    # print(response.content)