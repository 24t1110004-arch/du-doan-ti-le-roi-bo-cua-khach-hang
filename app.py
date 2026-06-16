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

st.set_page_config(page_title="Phân Tích Churn Khách Hàng Ngân Hàng", layout="wide")
st.title("📊 Ứng dụng Học máy: Dự đoán Khách hàng rời bỏ Ngân hàng (Churn Prediction)")
st.write("Ứng dụng chạy trực tiếp trên bộ dữ liệu thực tế **Churn_Modelling.csv** (10.000 dòng) và so sánh 4 thuật toán.")

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_Modelling.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("Không tìm thấy file 'Churn_Modelling.csv'. Vui lòng đảm bảo file này nằm cùng thư mục với file 'app.py'!")
    st.stop()

df_clean = df_raw.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

X_raw = df_clean.drop(columns=['Exited'])
y = df_clean['Exited']

X_encoded = pd.get_dummies(X_raw, columns=['Geography', 'Gender'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
}

accuracies = {}
for name, model in models.items():
    if name in ["K-Nearest Neighbors (KNN)", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    accuracies[name] = accuracy_score(y_test, y_pred)

tab1, tab2, tab3 = st.tabs(["📥 Input Dataset & So sánh Accuracy", "📈 5 Biểu đồ Trực quan", "📋 10 Dữ liệu Đầu tiên"])

with tab1:
    st.header("1. Tổng quan Bộ dữ liệu Churn Modelling")
    st.write(f"Bộ dữ liệu khổng lồ gồm có **{df_raw.shape[0]}** dòng và **{df_raw.shape[1]}** cột dữ liệu gốc.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Thông số mô tả dữ liệu (Các biến số):")
        numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        st.dataframe(df_raw[numeric_cols].describe().T[['mean', 'min', 'max']])
        
    with col2:
        st.subheader("🎯 Kết quả so sánh Độ chính xác (Accuracy):")
        acc_df = pd.DataFrame(list(accuracies.items()), columns=['Thuật toán', 'Accuracy'])
        acc_df['Accuracy'] = acc_df['Accuracy'].map(lambda x: f"{x*100:.2f}%")
        st.table(acc_df)
        
        best_model = max(accuracies, key=accuracies.get)
        st.success(f"🏆 Thuật toán đạt kết quả cao nhất: **{best_model}** ({accuracies[best_model]*100:.2f}%)")

with tab2:
    st.header("2. Trực quan hóa dữ liệu phân tích")
    
    st.subheader("Biểu đồ 1: Tỷ lệ khách hàng Ở lại (0) và Rời bỏ ngân hàng (1)")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    df_raw['Exited'].value_counts().plot.pie(labels=['Ở lại (0)', 'Rời bỏ (1)'], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], ax=ax1, startangle=90)
    ax1.set_ylabel('')
    st.pyplot(fig1)
    st.markdown("---")
    
    st.subheader("Biểu đồ 2: Phân bố Tuổi (Age) giữa hai nhóm khách hàng")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.boxplot(x='Exited', y='Age', data=df_raw, palette='Set2', ax=ax2)
    ax2.set_xticklabels(['Ở lại (0)', 'Rời bỏ (1)'])
    st.pyplot(fig2)
    st.markdown("---")

    st.subheader("Biểu đồ 3: Mối quan hệ giữa Điểm tín dụng và Số dư tài khoản")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='CreditScore', y='Balance', hue='Exited', data=df_raw.sample(2000, random_state=42), palette='coolwarm', alpha=0.5, ax=ax3)
    st.pyplot(fig3)
    st.markdown("---")

    st.subheader("Biểu đồ 4: Số lượng khách hàng rời bỏ theo Quốc gia (Geography)")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df_raw, x='Geography', hue='Exited', palette='muted', ax=ax4)
    st.pyplot(fig4)
    st.markdown("---")

    st.subheader("Biểu đồ 5: Ma trận tương quan giữa các đặc trưng số")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_clean[numeric_cols + ['Exited']].corr(), annot=True, cmap='RdBu', fmt='.2f', ax=ax5)
    st.pyplot(fig5)

with tab3:
    st.header("3. Chi tiết 10 dòng dữ liệu gốc đầu tiên từ Dataset")
    st.write("Dưới đây là 10 dòng dữ liệu thực tế mẫu trích xuất từ file CSV của bạn:")
    st.dataframe(df_raw.head(10))