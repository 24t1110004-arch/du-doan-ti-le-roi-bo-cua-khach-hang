import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Hệ Thống Dự Báo Churn Khách Hàng Ngân Hàng", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 38px !important;
            font-weight: 700 !important;
            color: #1E3A8A;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 16px !important;
            color: #4B5563;
            margin-bottom: 25px;
        }
        [data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 600 !important;
            color: #2563EB !important;
        }
        div[data-testid="metric-container"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 15px 20px !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }
        [data-testid="stSidebar"] {
            background-color: #1E293B !important;
        }
        [data-testid="stSidebar"] .stMarkdown h1, 
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] label {
            color: #F8FAFC !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'Churn_Modelling.csv'. Vui lòng đặt file này cùng thư mục với 'app.py'!")
    st.stop()

df_clean = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

X_raw = df_clean.drop(columns=['Exited'])
y = df_clean['Exited']

X_encoded = pd.get_dummies(X_raw, columns=['Geography', 'Gender'], drop_first=True)
all_features = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

@st.cache_resource
def train_models(_X_train, _X_train_scaled, _y_train):
    trained = {
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5).fit(_X_train_scaled, _y_train),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10).fit(_X_train, _y_train),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000).fit(_X_train_scaled, _y_train),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10).fit(_X_train, _y_train)
    }
    return trained

trained_models = train_models(X_train, X_train_scaled, y_train)

accuracies = {}
for name, model in trained_models.items():
    if name in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

st.sidebar.markdown("# ⚙️ BẢNG ĐIỀU KHIỂN")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Lựa chọn Thuật toán Dự báo")
selected_algorithm = st.sidebar.selectbox(
    "Mô hình huấn luyện áp dụng:",
    list(trained_models.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Thông tin Khách hàng")

input_credit = st.sidebar.slider("Điểm tín dụng (Credit Score)", 300, 850, 650)
input_geo = st.sidebar.selectbox("Quốc gia (Geography)", ["France", "Germany", "Spain"])
input_gender = st.sidebar.selectbox("Giới tính (Gender)", ["Female", "Male"])
input_age = st.sidebar.slider("Tuổi (Age)", 18, 92, 40)
input_tenure = st.sidebar.slider("Số năm gắn bó (Tenure)", 0, 10, 5)
input_balance = st.sidebar.number_input("Số dư tài khoản (EUR)", min_value=0.0, value=50000.0, step=1000.0)
input_products = st.sidebar.slider("Số sản phẩm ngân hàng đang dùng", 1, 4, 2)
input_card = st.sidebar.selectbox("Có thẻ tín dụng hay không?", ["Có", "Không"])
input_active = st.sidebar.selectbox("Thành viên hoạt động tích cực?", ["Có", "Không"])
input_salary = st.sidebar.number_input("Thu nhập ước tính/Năm (EUR)", min_value=0.0, value=100000.0, step=5000.0)

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 KÍCH HOẠT DỰ BÁO HỆ THỐNG", use_container_width=True)

st.markdown('<div class="main-title">💼 Dashboard Phân Tích & Dự Báo Khách Hàng Rời Bỏ Ngân Hàng</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Môi trường phân tích rủi ro tài chính chuyên sâu ứng dụng mô hình học máy có giám sát.</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📥 Công Cụ Dự Báo & Đối Soánh Mô Hình", 
    "📈 Trung Tâm Trực Quan Hóa Dữ Liệu", 
    "📋 Cơ Sở Dữ Liệu Thực Tế (Top 10)"
])

with tab1:
    st.subheader("🤖 Kết Quả Kiểm Định Trạng Thái Khách Hàng")
    
    if predict_button:
        input_data = pd.DataFrame(0, index=[0], columns=all_features)
        input_data.at[0, 'CreditScore'] = input_credit
        input_data.at[0, 'Age'] = input_age
        input_data.at[0, 'Tenure'] = input_tenure
        input_data.at[0, 'Balance'] = input_balance
        input_data.at[0, 'NumOfProducts'] = input_products
        input_data.at[0, 'HasCrCard'] = 1 if input_card == "Có" else 0
        input_data.at[0, 'IsActiveMember'] = 1 if input_active == "Có" else 0
        input_data.at[0, 'EstimatedSalary'] = input_salary
        
        if input_geo == "Germany":
            input_data.at[0, 'Geography_Germany'] = 1
        elif input_geo == "Spain":
            input_data.at[0, 'Geography_Spain'] = 1
            
        if input_gender == "Male":
            input_data.at[0, 'Gender_Male'] = 1

        current_model = trained_models[selected_algorithm]
        if selected_algorithm in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
            input_scaled = scaler.transform(input_data)
            prediction = current_model.predict(input_scaled)[0]
            try:
                prob = current_model.predict_proba(input_scaled)[0][1] * 100
            except:
                prob = None
        else:
            prediction = current_model.predict(input_data)[0]
            try:
                prob = current_model.predict_proba(input_data)[0][1] * 100
            except:
                prob = None

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label="Mô hình đang chạy", value=selected_algorithm.split(" (")[0])
        with res_col2:
            st.metric(label="Độ chính xác mẫu thử (Accuracy)", value=f"{accuracies[selected_algorithm]*100:.2f}%")
        with res_col3:
            if prob is not None:
                st.metric(label="Xác suất rủi ro Churn", value=f"{prob:.1f}%")
            else:
                st.metric(label="Trạng thái phân tích", value="Xác định lớp")

        st.markdown("#### **Kết luận từ mô hình hệ thống:**")
        if prediction == 1:
            st.error(f"⚠️ **CẢNH BÁO RỦI RO:** Khách hàng này có nguy cơ cao sẽ **RỜI BỎ** dịch vụ của ngân hàng trong tương lai gần. Đề xuất bộ phận vận hành triển khai chương trình CSKH đặc biệt.")
        else:
            st.success(f"✅ **CHỈ SỐ AN TOÀN:** Khách hàng có xu hướng tiếp tục **Ở LẠI** gắn bó lâu dài. Hồ sơ tài chính hoạt động ổn định.")
    else:
        st.info("💡 **HƯỚNG DẪN SỬ DỤNG:** Cấu hình các thông số hồ sơ khách hàng tại bảng điều khiển bên trái, sau đó nhấn nút **'KÍCH HOẠT DỰ BÁO HỆ THỐNG'** để xem kết quả phân tích rủi ro.")

    st.markdown("---")
    st.subheader("📊 Bảng Đối Soánh Hiệu Năng Toàn Diện Toàn Hệ Thống")
    
    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Thuật toán phân loại', 'Độ chính xác (Accuracy Score)'])
    acc_df['Độ chính xác (Accuracy Score)'] = acc_df['Độ chính xác (Accuracy Score)'].map(lambda x: f"{x*100:.2f}%")
    
    col_table, col_summary = st.columns([2, 1.2])
    with col_table:
        st.table(acc_df)
    with col_summary:
        best_model = max(accuracies, key=accuracies.get)
        st.info(
            f"**Nhận xét Học thuật:**\n\n"
            f"Mô hình đạt hiệu năng kiểm thử cao nhất trên tập dữ liệu là **{best_model}** "
            f"đạt tỷ lệ phân loại đúng **{accuracies[best_model]*100:.2f}%**. "
            f"Nhóm thuật toán Ensemble (Random Forest/Decision Tree) thường xử lý dữ liệu dạng bảng rất tối ưu."
        )

with tab2:
    st.header("📈 Trung Tâm Trực Quan Hóa Cấu Trúc Dữ Liệu Hệ Thống")
    numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown("##### **Biểu đồ 1: Tỷ lệ phân phối Tổng thể khách hàng**")
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        df_raw['Exited'].value_counts().plot.pie(
            labels=['Ở lại (0)', 'Rời bỏ (1)'], 
            autopct='%1.1f%%', 
            colors=['#2ecc71', '#e74c3c'], 
            ax=ax1, 
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='w')
        )
        ax1.set_ylabel('')
        st.pyplot(fig1)
        
    with row1_col2:
        st.markdown("##### **Biểu đồ 2: Phân tích biên độ Tuổi tác (Age)**")
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        sns.boxplot(x='Exited', y='Age', data=df_raw, palette='Set2', ax=ax2)
        ax2.set_xticklabels(['Ở lại (0)', 'Rời bỏ (1)'])
        ax2.set_xlabel('Trạng thái')
        ax2.set_ylabel('Tuổi')
        st.pyplot(fig2)
        
    st.markdown("---")
    
    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown("##### **Biểu đồ 3: Biến động hành vi theo không gian địa lý**")
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        sns.countplot(data=df_raw, x='Geography', hue='Exited', palette='viridis', ax=ax4)
        ax4.set_xlabel('Quốc gia')
        ax4.set_ylabel('Số lượng khách hàng')
        ax4.legend(['Ở lại (0)', 'Rời bỏ (1)'])
        st.pyplot(fig4)

    with row2_col2:
        st.markdown("##### **Biểu đồ 4: Ma trận tương quan đa biến (Correlation)**")
        fig5, ax5 = plt.subplots(figsize=(6, 3.8))
        sns.heatmap(df_clean[numeric_cols + ['Exited']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax5, cbar=False)
        st.pyplot(fig5)

    st.markdown("---")
    
    st.markdown("##### **Biểu đồ 5: Đồ thị phân tán đa chiều (Điểm tín dụng so với Số dư khả dụng)**")
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.scatterplot(x='CreditScore', y='Balance', hue='Exited', data=df_raw.sample(2000, random_state=42), palette='bwr', alpha=0.4, ax=ax3)
    ax3.set_xlabel('Điểm tín dụng (Credit Score)')
    ax3.set_ylabel('Số dư tài khoản (Balance)')
    ax3.legend(['Ở lại (0)', 'Rời bỏ (1)'])
    st.pyplot(fig3)

with tab3:
    st.header("📋 Bản ghi chi tiết Hệ thống Dữ liệu Đầu vào")
    st.write(f"Trích xuất hồ sơ 10 khách hàng đầu tiên của tổng thể bộ dữ liệu lớn ({df_raw.shape[0]} dòng):")
    st.dataframe(df_raw.head(10).style.background_gradient(cmap='YlOrRd', subset=['Balance', 'EstimatedSalary']))
