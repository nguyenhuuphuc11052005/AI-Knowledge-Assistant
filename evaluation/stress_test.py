import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import asyncio
import aiohttp
import time
import pandas as pd

API_URL = "http://localhost:8000/api/v1/chat"
TOTAL_REQUESTS = 100

SAMPLE_QUESTIONS = [
    "What is the policy for health insurance?",
    "Can you tell me about medical insurance policies?", 
    "How many vacation and personal days do employees get?",
    "What are the summer work hours at 37signals?",
    "How much can employees expense for a home office setup?",
    "Tell me the policy for health insurance again", 
]

# ĐÃ GỠ BỎ SEMAPHORE
async def send_single_request(session, request_id):
    query = SAMPLE_QUESTIONS[request_id % len(SAMPLE_QUESTIONS)]
    payload = {
        "query": query,
        "user_id": 3,
        "user_role": "admin",
        "session_id": f"stress_test_session_{request_id}"
    }
    
    start_time = time.time()
    try:
        async with session.post(API_URL, json=payload, timeout=60) as response:
            status = response.status
            source = "Unknown"
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: "):
                    import json
                    data_json = json.loads(line[6:])
                    if data_json["type"] == "cache":
                        source = "Semantic Cache"
                        break
                    elif data_json["type"] == "llm":
                        source = "LLM Generation (Groq)"
                        
            return {
                "request_id": request_id,
                "status": status,
                "latency_seconds": time.time() - start_time,
                "source": source,
                "success": True if status == 200 else False
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "ERROR",
            "latency_seconds": time.time() - start_time,
            "source": str(e),
            "success": False
        }
async def main():
    print(f"🔥 BẮT ĐẦU GIẢ LẬP STRESS TEST: RUN {TOTAL_REQUESTS} REQUESTS CÙNG LÚC...")
    start_total_time = time.time()
    
    # Giới hạn tối đa kết nối để tránh bị OS chặn socket
    connector = aiohttp.TCPConnector(limit=TOTAL_REQUESTS)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        # Tạo danh sách 50 task chạy đồng thời
        tasks = [send_single_request(session, i) for i in range(TOTAL_REQUESTS)]
        
        # Kích hoạt tất cả 50 requests bắn đi cùng một thời điểm!
        results = await asyncio.gather(*tasks)
        
    total_execution_time = time.time() - start_total_time
    
    # --- PHÂN TÍCH VÀ IN BÁO CÁO KẾT QUẢ ---
    df = pd.DataFrame(results)
    
    success_requests = df[df["success"] == True]
    failed_requests = df[df["success"] == False]
    
    cache_hits = df[df["source"] == "Semantic Cache"].shape[0]
    llm_generations = df[df["source"] == "LLM Generation"].shape[0]

    print("\n" + "="*50)
    print("📊 BÁO CÁO THỬ NGHIỆM CHỊU TẢI (STRESS TEST REPORT)")
    print("="*50)
    print(f" Tổng thời gian hoàn thành toàn bộ test: {total_execution_time:.2f} giây")
    print(f" Thống kê Request: Thành công {len(success_requests)}/{TOTAL_REQUESTS} | Thất bại {len(failed_requests)}")
    
    if not success_requests.empty:
        print("-"*50)
        print("⏱️ THÔNG THÁI VỀ THỜI GIAN PHẢN HỒI (LATENCY):")
        print(f"  - Nhanh nhất (Min Latency):  {success_requests['latency_seconds'].min():.4f} giây")
        print(f"  - Chậm nhất (Max Latency):  {success_requests['latency_seconds'].max():.4f} giây")
        print(f"  - Trung bình (Mean Latency): {success_requests['latency_seconds'].mean():.4f} giây")
        print(f"  - Phân vị 95% (95th Percentile - P95): {np.percentile(success_requests['latency_seconds'], 95):.4f} giây")
        
        print("-"*50)
        print("⚡ HIỆU QUẢ CỦA HỆ THỐNG CACHE:")
        print(f"  - Lấy thẳng từ Cache (Tốc độ mili-giây): {cache_hits} requests ({(cache_hits/TOTAL_REQUESTS)*100:.1f}%)")
        print(f"  - Phải gọi Llama 3.2 sinh chữ: {llm_generations} requests ({(llm_generations/TOTAL_REQUESTS)*100:.1f}%)")
    print("="*50)
    
    # Xuất ra file csv để lưu làm Portfolio
    df.to_csv("evaluation/stress_test_report.csv", index=False)
    print("📁 Chi tiết từng request đã được lưu tại: evaluation/stress_test_report.csv")

if __name__ == "__main__":
    asyncio.run(main())