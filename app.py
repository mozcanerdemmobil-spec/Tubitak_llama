import streamlit as st
from langchain_groq import ChatGroq

# 1. Sayfa Ayarları
st.set_page_config(page_title="Yapay Zeka Asistanım", page_icon="🤖")
st.title("💬 Groq & Langchain Chatbot")

# 2. Sidebar - API Anahtarı Girişi
with st.sidebar:
    st.header("Ayarlar")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.info("Anahtarınızı https://console.groq.com/ adresinden alabilirsiniz.")

# 3. Anahtar Kontrolü
if groq_api_key:
    try:
        # Daha kararlı ve hızlı olan llama3 modelini deniyoruz
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.1-8b-instant" 
        )
        
        # 4. Kullanıcı Girişi (text_input yerine chat_input daha akıcıdır)
        user_input = st.chat_input("Sorunuzu buraya yazın...")

        if user_input:
            # Mesajı ekranda göster
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.chat_message("assistant"):
                with st.spinner("Düşünüyorum..."):
                    # Groq'a gönder ve yanıtı al
                    response = llm.invoke(user_input)
                    st.write(response.content)
                    
    except Exception as e:
        # Hata türünü ekrana daha net basalım ki anlayalım
        st.error(f"Groq Bağlantı Hatası: {str(e)}")
else:
    st.warning("⚠️ Devam etmek için lütfen sol menüden Groq API anahtarınızı girin.")
    st.stop() # Anahtar yoksa kodun geri kalanını çalıştırmaz, böylece hata almazsın.
