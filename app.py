import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Judul Aplikasi
st.title('Regresi Linear untuk Data Produksi di tahun 1993-2015')

# Deskripsi Proyek
st.markdown("""
Aplikasi ini menggunakan regresi linear untuk memprediksi produksi berdasarkan data dari tahun 1993 hingga 2015.
Anda dapat mengunggah file CSV dengan format yang sesuai, melihat data, dan mengevaluasi model regresi linear.
Data yang digunakan berasal dari Kaggle.
""")

# Upload file CSV
uploaded_file = st.file_uploader('Unggah file CSV', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    file_name = uploaded_file.name
    
    # Tampilkan lima baris pertama dari data
    st.write('Lima baris pertama dari data:')
    st.write(data.head())
    
    # Tampilkan statistik deskriptif dari data
    st.write('Statistik Deskriptif:')
    st.write(data.describe())
    
    # Mengubah kolom dari 1993 hingga 2015 menjadi numerik
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Visualisasi tren produksi dari tahun 1993 hingga 2015
    st.write(f'Tren Produksi {file_name} (1993-2015):')
    data_melted = data.melt(id_vars=['Provinsi'], var_name='Tahun', value_name='Produksi')
    fig, ax = plt.subplots(figsize=(10, 6))
    for provinsi in data['Provinsi'].unique():
        provinsi_data = data_melted[data_melted['Provinsi'] == provinsi]
        ax.plot(provinsi_data['Tahun'], provinsi_data['Produksi'], label=provinsi)
    
    ax.set_xlabel('Tahun')
    ax.set_ylabel('Produksi')
    ax.set_title('Tren Produksi Padi')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True)
    st.pyplot(fig)
    
    # Memilih fitur dan target
    X = data.drop(['Provinsi', '2015'], axis=1)
    y = data['2015']
    
    # Mengisi nilai NaN dengan median kolom
    X = X.fillna(X.median())
    
    # Membagi data (80% pelatihan, 20% pengujian)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat model regresi linear
    model = LinearRegression()
    
    # Melatih model
    model.fit(X_train, y_train)
    
    # Prediksi menggunakan data pengujian
    y_pred = model.predict(X_test)
    
    # Menghitung MSE dan R²
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Tampilkan hasil evaluasi model
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'R² Score: {r2}')
    
    # Plot prediksi vs nilai sebenarnya
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, c='blue', label='Prediksi')
    ax.scatter(y_test, y_test, c='red', label='Nilai Sebenarnya', alpha=0.5)
    ax.set_xlabel('Nilai Sebenarnya')
    ax.set_ylabel('Prediksi')
    ax.set_title('Prediksi vs Nilai Sebenarnya')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.write("Silakan unggah file CSV untuk memulai.")
