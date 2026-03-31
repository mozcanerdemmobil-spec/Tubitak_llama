import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Daha güncel kütüphane yolu

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓", layout="wide")

# CSS ile biraz daha şık bir görünüm
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    </style>
    """, unsafe_allow_index=True)

st.title("🎓 MEB Ortaöğretim Yönetmelik Asistanı")
st.info("Bu asistan, MEB Ortaöğretim Kurumları Yönetmeliği temel alınarak hazırlanmıştır.")

# --- 2. VERİ TABANI VE MODEL HAZIRLIĞI ---
@st.cache_resource
def load_data():
    try:
        # Embedding modelini yüklüyoruz
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # Veri tabanını klasörden yüklüyoruz
        vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Veri tabanı yüklenirken hata oluştu: {e}")
        return None

# Sidebar - Ayarlar ve Profil Seçimi
with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Groq API Key", type="password", help="Groq Cloud üzerinden aldığınız API anahtarını girin.").strip()
    
    if not api_key:
        st.warning("⚠️ Devam etmek için lütfen API Key giriniz.")
        st.stop()

    st.divider()
    
    secenekler = [
        "Görevli", "Öğrenci 9-A/BL", "Öğrenci 9-B/BL", "Öğrenci 9-C/BL", "Öğrenci 9-D/BL", 
        "Öğrenci 9-E/EL", "Öğrenci 9-F/EL", "Öğrenci 9-G/EL", "Öğrenci 9-H/EL", "Öğrenci 9-I/EL", 
        "Öğrenci 9/ATP", "Öğrenci 10A/BL", "Öğrenci 10B/BL", "Öğrenci 10C/EN", "Öğrenci 10D/HB", 
        "Öğrenci 10E/HB", "Öğrenci 10 ATP", "Öğrenci 11A/BL", "Öğrenci 11B/BL", "Öğrenci 11C/EN", 
        "Öğrenci 11D/HB", "Öğrenci 11 ATP", "Öğrenci 12A/BL", "Öğrenci 12B/BL", "Öğrenci 12F/HB", 
        "Öğrenci 12G/EN", "Öğrenci 12 ATP"
    ]
    secilen_rol = st.selectbox("Lütfen rolünüzü seçiniz:", secenekler)
    st.success(f"Profil: **{secilen_rol}**")

# Gerekli nesneleri oluşturuyoruz
client = Groq(api_key=api_key)
vector_db = load_data()

# --- 3. SORGULAMA FONKSİYONU ---
def okul_asistani_sorgula(soru):
    if vector_db is None:
        return "Hata: Veri tabanı yüklenemedi."

    # Benzerlik araması yaparak bağlamı çekiyoruz
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # Sistem Promptu (Senin kurallarınla optimize edildi)
    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzman, teknik ve resmi bir asistansın.
    Kullanıcı Rolü: {secilen_rol}

    Kritik Kurallar:
    1. SADECE sana verilen 'Bağlam' ve 'İkincil Kaynak' içindeki bilgileri kullan.
    2. Cevap bağlamda yoksa, 'Bu konuyla ilgili yönetmelikte net bir bilgi bulamadım' de. Dış bilgi ekleme.
    3. Cevaplarını maddeler halinde ve resmi bir dille ver.
    4. Cevap formatı SADECE şu olmalı:
    Cevap:
    - (İçerik)

    5. "Ek Bilgi" başlığı kullanma, cevabın içine yedir.
    6. "Geçer miyim?", "Yapabilir miyim?" gibi sorulara mutlaka "Evet" veya "Hayır" ile başla.
    7. Sayısal verilerde (50 puan, 5 gün devamsızlık vb.) net ol.
    8. Bağlam ve İkincil Kaynak çelişirse, Bağlamı esas al.

    İkincil Kaynak Bilgileri:
    - 5 kez derse geç kalma yarım gün devamsızlık sayılır.
    - DYK mazeretsiz devamsızlık sınırı %20'dir.
    - Teşekkür için en az 70, Takdir için en az 85 puan gerekir.
    - 5 gün özürsüz devamsızlık belge alımına engeldir.
    - Yıl sonu başarı puanı 50 ve üzeri ise sınıf geçilir (sorumlu geçiş dahil).
    - Baraj derslerden (Edebiyat vb.) kalınca ortalama 50 üstü olsa bile sorumlu geçilir.

    Bağlam:
    {baglam}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Soru: {soru}"}
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=1000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Sorgu sırasında bir hata oluştu: {str(e)}"

# --- 4. ARAYÜZ (SOHBET GEÇMİŞİ) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Kullanıcı girişi
if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Yönetmelik inceleniyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})

st.divider()
st.caption("⚠️ **Not:** Bu asistan yapay zeka ile çalışmaktadır. Resmi işlemlerinizde e-Okul ve okul idaresinin verilerini esas alınız.")
