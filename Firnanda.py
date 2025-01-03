import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model, encoder, dan scaler
def load_model_and_preprocessors(model_name):
    if model_name == "CatBoost":
        model = pickle.load(open("catboost.pkl", "rb"))
        encoder = pickle.load(open("enc_cb.pkl", "rb"))
        scaler = pickle.load(open("scaler_cb.pkl", "rb"))
    elif model_name == "Perceptron":
        model = pickle.load(open("ppn_model.pkl", "rb"))
        encoder = pickle.load(open("enc_ppn.pkl", "rb"))
        scaler = pickle.load(open("sc_ppn.pkl", "rb"))
    else:
        raise ValueError("Model tidak ditemukan!")
    return model, encoder, scaler


# 🎯 Judul Aplikasi
st.title(" Prediksi Jenis Ikan Ajaib!")
st.write("Masukkan data ikannya, pilih algoritma super, dan lihat jenis ikannya. Siap-siap kaget! ")


# 🧠 Pilih Algoritma Canggih
st.write("###  Pilih Mesin Prediksi Ajaib")
algorithm = st.selectbox(
    " Pilih Algoritma Ramalan Ikan:",
    ["CatBoost", "Perceptron"]
)


# 🐠 Input Data Ikan (Menggunakan Kolom untuk Desain Lebih Rapi)
st.write("### Maukkan Data Ikanmu!")

# Menggunakan kolom untuk menyusun input agar lebih menarik
col1, col2, col3 = st.columns(3)

with col1:
    feature1 = st.number_input(" **Panjang Ikan (cm)**", min_value=0.0, format="%.2f", help="Masukkan panjang ikan dalam cm")

with col2:
    feature2 = st.number_input(" **Berat Ikan (kg)**", min_value=0.0, format="%.2f", help="Masukkan berat ikan dalam kg")

with col3:
    feature3 = st.number_input(" **Rasio Berat-Panjang (kg/cm)**", min_value=0.0, format="%.2f", help="Berat dibagi panjang ikan")


# 🚀 Tombol untuk Prediksi
st.write("---")  # Garis pemisah untuk estetika
st.write("###  **Siap untuk Tebak Jenis Ikan?**")

# Tombol prediksi dengan styling lebih menarik
if st.button(" **Prediksi Sekarang!**"):
    try:
        # Load model, encoder, dan scaler berdasarkan algoritma yang dipilih
        model, encoder, scaler = load_model_and_preprocessors(algorithm)
        
        # Scaling data input
        input_data = np.array([[feature1, feature2, feature3]])
        input_data_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_data_scaled)
        prediction_label = encoder.inverse_transform(prediction)
        
        # 🥳 Tampilkan Hasil Prediksi
        st.success(f" **Jenis ikannya adalah: {prediction_label[0]}** ")
        st.balloons()
    except Exception as e:
        st.error(f" Ups! Ada yang salah: {e}")


# 🎨 Sentuhan Akhir dengan Styling CSS
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border-radius: 8px;
    }
    .stNumberInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    .stSelectbox>div>div>select {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
