import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import time

# --- Cấu hình API Gemini ---
# Sử dụng mô hình gemini-2.5-flash-preview-05-20 cho các tác vụ trích xuất và phân tích.
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
API_KEY = "" # API_KEY sẽ được tự động cung cấp trong môi trường Canvas
MAX_RETRIES = 5

# --- Hàm hỗ trợ cho việc gọi API (Bao gồm Exponential Backoff) ---

def call_gemini_api(system_prompt, user_query, response_schema=None):
    """Gửi yêu cầu đến Gemini API với cơ chế exponential backoff."""
    headers = {'Content-Type': 'application/json'}
    
    # Payload cơ bản
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]}
    }

    # Thêm cấu hình schema nếu cần structured output
    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(API_URL + API_KEY, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                result = response.json()
                # Trích xuất nội dung text từ response
                candidate = result.get('candidates', [{}])[0]
                text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
                return text
            
            # Xử lý lỗi Rate Limit hoặc lỗi 500
            if response.status_code == 429 or response.status_code >= 500:
                st.warning(f"Lỗi API (HTTP {response.status_code}). Đang thử lại sau {2**attempt} giây...")
                time.sleep(2**attempt)
            else:
                st.error(f"Lỗi API không mong muốn (HTTP {response.status_code}): {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                st.warning(f"Lỗi kết nối: {e}. Đang thử lại sau {2**attempt} giây...")
                time.sleep(2**attempt)
            else:
                st.error(f"Lỗi kết nối sau {MAX_RETRIES} lần thử: {e}")
                return None
    return None

# --- Định nghĩa Schema cho Trích xuất Dữ liệu Cấu trúc ---
EXTRACTION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "Vốn_Đầu_Tư_VND": {"type": "NUMBER", "description": "Vốn đầu tư ban đầu, đơn vị: VND. Chỉ lấy phần số."},
        "Dòng_Đời_Dự_Án_Năm": {"type": "NUMBER", "description": "Số năm dự án hoạt động. Chỉ lấy phần số."},
        "Doanh_Thu_Hàng_Năm_VND": {"type": "NUMBER", "description": "Doanh thu hàng năm, đơn vị: VND. Chỉ lấy phần số."},
        "Chi_Phí_Hàng_Năm_VND": {"type": "NUMBER", "description": "Chi phí vận hành hàng năm (không bao gồm lãi vay và khấu hao), đơn vị: VND. Chỉ lấy phần số."},
        "WACC_Phần_Trăm": {"type": "NUMBER", "description": "Chi phí vốn bình quân (WACC), đơn vị: phần trăm (%). Ví dụ: 13% là 0.13. Chỉ lấy phần số thập phân."},
        "Thuế_Suất_Phần_Trăm": {"type": "NUMBER", "description": "Thuế suất thu nhập doanh nghiệp, đơn vị: phần trăm (%). Ví dụ: 20% là 0.2. Chỉ lấy phần số thập phân."}
    },
    "required": ["Vốn_Đầu_Tư_VND", "Dòng_Đời_Dự_Án_Năm", "Doanh_Thu_Hàng_Năm_VND", "Chi_Phí_Hàng_Năm_VND", "WACC_Phần_Trăm", "Thuế_Suất_Phần_Trăm"]
}

# --- Chức năng 1: Trích xuất Dữ liệu Tài chính ---
def extract_financial_data(text_content):
    """Sử dụng AI để trích xuất dữ liệu tài chính từ nội dung văn bản."""
    system_prompt = (
        "Bạn là một Trợ lý AI chuyên nghiệp về tài chính dự án. Nhiệm vụ của bạn là đọc "
        "văn bản đầu vào và trích xuất sáu thông số tài chính chính xác (Vốn đầu tư, "
        "Dòng đời dự án, Doanh thu, Chi phí, WACC, Thuế suất). "
        "Đầu ra phải là một đối tượng JSON TUYỆT ĐỐI KHÔNG CÓ LỜI DẪN, theo đúng schema "
        "và đơn vị đã định nghĩa (VND cho tiền tệ, số thập phân cho tỷ lệ %)."
    )
    
    st.info("Đang trích xuất dữ liệu tài chính từ văn bản...")
    
    json_str = call_gemini_api(system_prompt, text_content, EXTRACTION_SCHEMA)
    
    if json_str:
        try:
            # Xử lý các lỗi phổ biến của JSON đầu ra (ví dụ: có dấu xuống dòng)
            json_str = json_str.strip()
            data = json.loads(json_str)
            st.session_state.extracted_data = data
            st.success("Trích xuất dữ liệu thành công!")
            return data
        except json.JSONDecodeError as e:
            st.error(f"Lỗi giải mã JSON từ AI: {e}. Đầu ra nhận được: {json_str}")
            return None
    return None

# --- Chức năng 2 & 3: Xây dựng Dòng tiền & Tính toán Chỉ số ---
def calculate_financial_metrics(data):
    """Xây dựng bảng dòng tiền và tính toán NPV, IRR, PP, DPP."""
    try:
        # Lấy các thông số
        I0 = data['Vốn_Đầu_Tư_VND']
        N = int(data['Dòng_Đời_Dự_Án_Năm'])
        R = data['Doanh_Thu_Hàng_Năm_VND']
        C = data['Chi_Phí_Hàng_Năm_VND']
        WACC = data['WACC_Phần_Trăm']
        T = data['Thuế_Suất_Phần_Trăm']
        
        # 1. Tính toán Dòng tiền Thuần Hàng năm (Net Cash Flow - NCF)
        # Giả định: Thuế tính trên Lợi nhuận trước thuế (EBT = R - C)
        EBT = R - C
        Tax = EBT * T
        EAT = EBT - Tax # Lợi nhuận sau thuế
        # Giả định: Không có Khấu hao hoặc Khấu hao được bù trừ (NCF = EAT)
        # NCF = EAT + Khấu hao (nếu có)
        # Trong ví dụ này, do không có Khấu hao, ta lấy NCF = EAT
        NCF_annual = EAT 
        
        # 2. Xây dựng Bảng Dòng tiền (Cash Flow Table)
        
        # Dòng tiền cho từng năm
        years = np.arange(0, N + 1)
        cash_flows = np.zeros(N + 1)
        cash_flows[0] = -I0  # Vốn đầu tư ban đầu (năm 0)
        cash_flows[1:] = NCF_annual # Dòng tiền thuần hàng năm
        
        # Hệ số Chiết khấu (Discount Factor)
        discount_factors = 1 / (1 + WACC)**years
        
        # Dòng tiền Chiết khấu (Discounted Cash Flow - DCF)
        dcf = cash_flows * discount_factors
        
        # Dòng tiền Tích lũy và Dòng tiền Chiết khấu Tích lũy
        cumulative_cf = np.cumsum(cash_flows)
        cumulative_dcf = np.cumsum(dcf)
        
        # Tạo DataFrame cho Bảng Dòng tiền
        df_cf = pd.DataFrame({
            'Năm (t)': years,
            'Hệ số Chiết khấu': [f"{df:.4f}" for df in discount_factors],
            'Dòng tiền (CFt) (VND)': [f"{cf:,.0f}" for cf in cash_flows],
            'Dòng tiền Chiết khấu (DCFt) (VND)': [f"{dcf_val:,.0f}" for dcf_val in dcf],
            'Dòng tiền Tích lũy (VND)': [f"{ccf:,.0f}" for ccf in cumulative_cf],
            'Dòng tiền Chiết khấu Tích lũy (VND)': [f"{cdcf:,.0f}" for cdcf in cumulative_dcf]
        })
        
        # 3. Tính toán các chỉ số
        
        # a. NPV (Net Present Value)
        npv = np.sum(dcf)
        
        # b. IRR (Internal Rate of Return)
        # np.irr hoạt động tốt nhất nếu CF đầu tiên âm và các CF sau dương
        irr = np.irr(cash_flows)
        
        # c. PP (Payback Period - Thời gian hoàn vốn)
        # Tìm năm đầu tiên mà Dòng tiền Tích lũy >= 0
        pp = next((t for t, ccf in enumerate(cumulative_cf) if ccf >= 0), N)
        if pp < N:
            # Nội suy (Interpolation)
            cf_before = cumulative_cf[pp-1]
            cf_at = cash_flows[pp]
            pp = pp - 1 + abs(cf_before) / cf_at
        
        # d. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
        # Tìm năm đầu tiên mà Dòng tiền Chiết khấu Tích lũy >= 0
        dpp = next((t for t, cdcf in enumerate(cumulative_dcf) if cdcf >= 0), N)
        if dpp < N:
            # Nội suy (Interpolation)
            dcf_before = cumulative_dcf[dpp-1]
            dcf_at = dcf[dpp]
            dpp = dpp - 1 + abs(dcf_before) / dcf_at
            
        metrics = {
            "NPV": npv,
            "IRR": irr,
            "PP": pp,
            "DPP": dpp
        }
        
        st.session_state.cash_flow_df = df_cf
        st.session_state.financial_metrics = metrics
        
        return df_cf, metrics
        
    except Exception as e:
        st.error(f"Lỗi trong quá trình tính toán tài chính: {e}")
        return None, None

# --- Chức năng 4: Phân tích các Chỉ số Hiệu quả ---
def analyze_metrics_ai(data, metrics):
    """Yêu cầu AI phân tích các chỉ số hiệu quả dự án."""
    
    # Chuẩn bị dữ liệu cho AI
    input_data_str = json.dumps(data, indent=2)
    metrics_str = (
        f"NPV (Giá trị hiện tại ròng): {metrics['NPV']:,.0f} VND\n"
        f"IRR (Tỷ suất sinh lời nội tại): {metrics['IRR']:.2%}\n"
        f"PP (Thời gian hoàn vốn): {metrics['PP']:.2f} năm\n"
        f"DPP (Thời gian hoàn vốn có chiết khấu): {metrics['DPP']:.2f} năm"
    )
    
    system_prompt = (
        "Bạn là một chuyên gia phân tích tài chính dự án cao cấp. "
        "Nhiệm vụ của bạn là phân tích tính hiệu quả của dự án dựa trên các chỉ số NPV, IRR, PP, và DPP được cung cấp. "
        "Đưa ra nhận định chi tiết, rõ ràng về:\n"
        "1. Tính khả thi của dự án (dựa trên NPV và IRR so với WACC).\n"
        "2. Khả năng thanh khoản (dựa trên PP và DPP so với Dòng đời dự án).\n"
        "3. Kết luận cuối cùng về việc chấp nhận hay từ chối đầu tư.\n"
        "Hãy viết bằng tiếng Việt, giọng điệu chuyên nghiệp, trình bày theo từng luận điểm."
    )
    
    user_query = (
        f"Dữ liệu đầu vào của dự án:\n{input_data_str}\n\n"
        f"Các chỉ số hiệu quả đã tính toán:\n{metrics_str}\n\n"
        "Hãy thực hiện phân tích chi tiết."
    )
    
    st.info("Đang yêu cầu AI phân tích các chỉ số tài chính...")
    
    analysis_result = call_gemini_api(system_prompt, user_query)
    
    if analysis_result:
        st.session_state.ai_analysis = analysis_result
        st.success("Phân tích AI hoàn tất!")

# --- Giao diện Streamlit Chính ---
def main():
    st.set_page_config(page_title="Phân Tích Dự Án Tài Chính Tự Động", layout="wide")
    st.title("💰 Ứng Dụng Đánh Giá Hiệu Quả Dự Án Tài Chính (AI-Powered)")
    st.markdown("---")

    # Khởi tạo session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'cash_flow_df' not in st.session_state:
        st.session_state.cash_flow_df = None
    if 'financial_metrics' not in st.session_state:
        st.session_state.financial_metrics = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
        
    st.sidebar.header("1. Nhập liệu Phương án Kinh doanh")
    
    # Hướng dẫn và ô nhập liệu
    st.sidebar.markdown(
        """
        **Do hạn chế về môi trường, vui lòng dán (paste) nội dung**
        **từ file Word của Phương án Đầu tư vào ô bên dưới để AI trích xuất.**
        
        *Ví dụ: Bạn có thể dán nội dung từ file Markdown đã tạo trước đó.*
        """
    )
    
    # Nội dung mẫu từ file Phương án Đầu tư Bánh mì trước đó
    sample_content = (
        "# PHƯƠNG ÁN ĐẦU TƯ DỰ ÁN DÂY CHUYỀN SẢN XUẤT BÁNH MÌ HIỆN ĐẠI\n"
        "Tổng Vốn Đầu tư: 30.000.000.000 VND. Vòng đời Dự án: 10 năm.\n"
        "Bắt đầu có dòng tiền từ cuối năm thứ nhất. WACC của doanh nghiệp là 13%.\n"
        "Doanh thu Hàng năm: 3.5 tỷ VND. Chi phí Vận hành Hàng năm: 2 tỷ VND.\n"
        "Thuế suất Thu nhập Doanh nghiệp: 20%."
    )
    
    document_content = st.sidebar.text_area(
        "Dán Nội dung Phương án Kinh doanh:", 
        value=sample_content,
        height=300,
        key="doc_content"
    )

    # Nút bấm để thực hiện trích xuất dữ liệu
    if st.sidebar.button("✨ Lọc Dữ liệu Tài chính (AI)"):
        if document_content:
            with st.spinner("Đang kết nối AI để trích xuất dữ liệu..."):
                extract_financial_data(document_content)
        else:
            st.sidebar.error("Vui lòng dán nội dung phương án kinh doanh vào ô nhập liệu.")

    # --- Hiển thị kết quả Trích xuất & Tính toán ---

    st.header("2. Dữ liệu Trích xuất và Tính toán Dòng tiền")
    
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        
        # Bảng tóm tắt dữ liệu trích xuất
        st.subheader("2.1. Dữ liệu Tài chính Đã Trích xuất")
        col1, col2, col3 = st.columns(3)
        
        # Hiển thị dữ liệu trích xuất dưới dạng thẻ
        col1.metric("Vốn Đầu tư (I0)", f"{data['Vốn_Đầu_Tư_VND']:,.0f} VND")
        col2.metric("Dòng Đời Dự Án (N)", f"{data['Dòng_Đời_Dự_Án_Năm']} năm")
        col3.metric("WACC (k)", f"{data['WACC_Phần_Trăm']:.2%}")
        
        col1.metric("Doanh Thu Hàng Năm (R)", f"{data['Doanh_Thu_Hàng_Năm_VND']:,.0f} VND")
        col2.metric("Chi Phí Hàng Năm (C)", f"{data['Chi_Phí_Hàng_Năm_VND']:,.0f} VND")
        col3.metric("Thuế Suất (T)", f"{data['Thuế_Suất_Phần_Trăm']:.0%}")
        
        # Nút tính toán dòng tiền
        if st.button("📊 Xây dựng Dòng tiền & Tính Chỉ số"):
            with st.spinner("Đang xây dựng bảng dòng tiền và tính toán các chỉ số..."):
                calculate_financial_metrics(data)
    
    if st.session_state.cash_flow_df is not None:
        st.subheader("2.2. Bảng Dòng tiền (Cash Flow)")
        st.dataframe(st.session_state.cash_flow_df, use_container_width=True)

        # --- Hiển thị các chỉ số Hiệu quả ---
        st.header("3. Các Chỉ số Đánh giá Hiệu quả Dự án")
        metrics = st.session_state.financial_metrics
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        col_m1.metric("NPV (Giá trị hiện tại ròng)", f"{metrics['NPV']:,.0f} VND", 
                      delta="Dự án Khả thi" if metrics['NPV'] > 0 else "Dự án Không khả thi")
        
        col_m2.metric("IRR (Tỷ suất sinh lời nội tại)", f"{metrics['IRR']:.2%}",
                      delta="Lớn hơn WACC" if metrics['IRR'] > data['WACC_Phần_Trăm'] else "Nhỏ hơn WACC")

        col_m3.metric("PP (Thời gian hoàn vốn)", f"{metrics['PP']:.2f} năm",
                      delta=f"<= {data['Dòng_Đời_Dự_Án_Năm']} năm")
        
        col_m4.metric("DPP (Hoàn vốn có chiết khấu)", f"{metrics['DPP']:.2f} năm",
                      delta=f"<= {data['Dòng_Đời_Dự_Án_Năm']} năm")

        st.markdown("---")
        
        # --- Chức năng 4: Phân tích AI ---
        st.header("4. Phân tích Chuyên sâu của AI")
        
        if st.session_state.ai_analysis:
            st.markdown(st.session_state.ai_analysis)
        
        if st.button("🧠 Yêu cầu AI Phân tích Hiệu quả Dự án"):
            with st.spinner("Đang tổng hợp dữ liệu và phân tích chuyên sâu..."):
                analyze_metrics_ai(data, metrics)
            # Sau khi phân tích xong, nội dung sẽ được cập nhật nhờ session_state

    elif st.session_state.extracted_data:
        st.info("Vui lòng nhấn nút 'Xây dựng Dòng tiền & Tính Chỉ số' để tiếp tục.")
    else:
        st.info("Vui lòng nhấn nút 'Lọc Dữ liệu Tài chính (AI)' ở sidebar để bắt đầu.")

if __name__ == "__main__":
    main()
