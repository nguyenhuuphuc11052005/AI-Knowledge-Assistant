import streamlit as st
import requests
import uuid

# Cấu hình API URL
API_URL = "http://localhost:8000/api/v1/chat"

st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🤖", layout="wide")

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.title("⚙️ Bảng Điều Khiển (Thin Client)")
    selected_role = st.selectbox("Chọn vai trò:", ["intern", "manager", "admin"])
    user_id = {"intern": 1, "manager": 2, "admin": 3}[selected_role]
    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.title("🏢 Enterprise AI Knowledge Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Bạn muốn hỏi gì về quy định công ty?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        def get_api_stream():
            payload = {
                "query": prompt,
                "user_id": 1, # Tạm thời fix cứng hoặc lấy từ sidebar của bạn
                "user_role": "admin",
                "session_id": st.session_state.session_id
            }
            # QUAN TRỌNG: Thêm tham số stream=True vào requests.post
            response = requests.post(API_URL, json=payload, stream=True)
            
            if response.status_code == 200:
                # Duyệt qua từng dòng/từng chunk dữ liệu đổ về từ API
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        yield chunk
            else:
                yield f"❌ Lỗi máy chủ: Không thể kết nối luồng (Code {response.status_code})"

        # Kích hoạt tính năng streaming trực quan của Streamlit
        # st.write_stream sẽ tự động lặp qua hàm get_api_stream, in chữ ra màn hình 
        # và trả về TOÀN BỘ câu văn hoàn chỉnh sau khi stream kết thúc.
        full_response = st.write_stream(get_api_stream())
        
        # Lưu câu trả lời hoàn chỉnh vào lịch sử chat để không bị mất khi reload
        st.session_state.messages.append({"role": "assistant", "content": full_response})