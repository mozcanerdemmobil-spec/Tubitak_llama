import streamlit as st
from langchain_groq import ChatGroq
# Diğer importlarını buraya ekle (FAISS, vb.)

# 1. Sayfa Ayarları
st.set_page_config(page_title="Yapay Zeka Asistanım", page_icon="🤖")
st.title("💬 Groq & Langchain Chatbot")

# 2. API Anahtarı ve Model Kurulumu
# Not: API anahtarını güvenli tutmak için st.secrets veya .env kullanmalısın
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    
    # 3. Kullanıcı Girişi
    user_input = st.text_input("Sorunuzu yazın:")

    if user_input:
        with st.spinner("Düşünüyorum..."):
            # Burada Langchain / FAISS akışını çalıştır
            response = llm.invoke(user_input)
            st.write("### Cevap:")
            st.success(response.content)
else:
    st.warning("Lütfen sol tarafa Groq API anahtarınızı girin.")
