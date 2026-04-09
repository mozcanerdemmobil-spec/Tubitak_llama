import streamlit as st
import pandas as pd
from groq import Groq
import re

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓", layout="wide")
st.title("🎓 MEB Ders Programı Asistanı")

# --- 2. DERS PROGRAMI YÜKLEME (DINAMIK) ---

@st.cache_data
def tum_programlari_yukle():
    """Excel'i bir kez indirir ve tüm sayfaları temiz isimlerle sözlüğe atar."""
    url = "https://raw.githubusercontent.com/mozcanerdemmobil-spec/Tubitak_llama/main/programlar.xlsx"
    program_sozlugu = {}
    try:
        # Excel dosyasını oku
        excel_file = pd.ExcelFile(url, engine='openpyxl')
        for sayfa_adi in excel_file.sheet_names:
            # "12-A " -> "12a" formatına getir
            temiz_isim = sayfa_adi.lower().replace(" ", "").replace("-", "")
            df = pd.read_excel(excel_file, sheet_name=sayfa_adi, engine='openpyxl')
            program_sozlugu[temiz_isim] = {
                "orijinal_isim": sayfa_adi,
                "veri": df
            }
        return program_sozlugu
    except Exception as e:
        st.error(f"Excel dosyası yüklenemedi: {e}")
        return None

# Programları belleğe al
programlar = tum_programlari_yukle()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()
    if not api_key:
        st.warning("Lütfen devam etmek için API Key giriniz.")
        st.stop()

# --- 4. GROQ BAĞLANTISI ---
client = Groq(api_key=api_key)

# --- 5. GENEL YÖNETMELİK BİLGİSİ (Sabit Prompt) ---
# Vector DB olmadığı için en kritik yönetmelik bilgilerini burada tutuyoruz.
YONETMELIK_OZET = """
Sen MEB Ortaöğretim Kurumları Yönetmeliği uzmanısın.
Kritik Bilgiler:
- Ders süresi 40 dk.
- Özürsüz devamsızlık sınırı 10 gün, toplam sınır 30 gün.
- Bir dersten başarılı sayılmak için puan en az 50 olmalı.
- Takdir belgesi 85.00+, Teşekkür 70.00-84.99 arası.
- Sorumlu geçme sınırı en fazla 3 ders.
"""

# --- 6. CHAT ARAYÜZÜ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın (Örn: 10A programı nedir?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Girdiyi temizle
    temiz_prompt = prompt.lower().replace(" ", "").replace("-", "")
    istenen_sinif_key = None

    # Sınıf tespiti
    if programlar:
        for sinif_key in programlar.keys():
            if sinif_key in temiz_prompt:
                istenen_sinif_key = sinif_key
                break

    with st.chat_message("assistant"):
        with st.spinner("Düşünüyorum..."):
            
            # DURUM A: Ders Programı Sorusu
            if istenen_sinif_key:
                sinif_verisi = programlar[istenen_sinif_key]
                program_csv = sinif_verisi["veri"].to_csv(index=False)
                
                ai_prompt = f"""
                Aşağıda {sinif_verisi['orijinal_isim']} sınıfının programı var:
                {program_csv}
                Soru: {prompt}
                Lütfen bu programa göre net bir cevap ver.
                """
                # Yanıt oluştur
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": ai_prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0
                )
            
            # DURUM B: Genel Yönetmelik Sorusu
            else:
                ai_prompt = f"{YONETMELIK_OZET}\nSoru: {prompt}\nCevabı maddeler halinde ver."
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": ai_prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0
                )

            cevap = response.choices[0].message.content
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.caption("⚠️ Bilgileri resmi kaynaklardan doğrulamayı unutmayın.")
