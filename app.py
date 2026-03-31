import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓")

# CSS Hatası düzeltildi: unsafe_allow_html=True yapıldı
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎓 MEB Ortaöğretim Yönetmelik Asistanı")

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
        st.error(f"Veri tabanı yüklenemedi: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("Ayarlar")
    api_key = st.text_input("Groq API Key", type="password").strip()
    
    if not api_key:
        st.warning("Lütfen devam etmek için API Key giriniz.")
        st.stop()

    secenekler = ["Görevli", "Öğrenci 9-A/BL", "Öğrenci 9-B/BL", "Öğrenci 9-C/BL", "Öğrenci 9-D/BL", "Öğrenci 9-E/EL", "Öğrenci 9-F/EL", "Öğrenci 9-G/EL", "Öğrenci 9-H/EL", "Öğrenci 9-I/EL", "Öğrenci 9/ATP", "Öğrenci 10A/BL", "Öğrenci 10B/BL", "Öğrenci 10C/EN", "Öğrenci 10D/HB", "Öğrenci 10E/HB", "Öğrenci 10 ATP", "Öğrenci 11A/BL", "Öğrenci 11B/BL", "Öğrenci 11C/EN", "Öğrenci 11D/HB", "Öğrenci 11 ATP", "Öğrenci 12A/BL", "Öğrenci 12B/BL", "Öğrenci 12F/HB", "Öğrenci 12G/EN", "Öğrenci 12 ATP"]
    secilen_rol = st.selectbox("Lütfen rolünüzü seçiniz:", secenekler)

st.write(f"Şu anki profil: **{secilen_rol}**")

# Gerekli nesneler
client = Groq(api_key=api_key)
vector_db = load_data()

# --- 3. SORGULAMA FONKSİYONU ---
def okul_asistani_sorgula(soru):
    if vector_db is None:
        return "Veri tabanı hazır değil."

    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzmansın.
    Kullanıcı Rolü: {secilen_rol}

    KURALLAR:
    1. Sadece bağlam ve ek bilgiyi kullan. Bilmiyorsan 'net bilgi bulamadım' de.
    2. Cevapların başında mutlaka "Cevap:" ifadesi olsun.
    3. "Geçer miyim?" gibi sorulara Evet/Hayır ile başla.
    4. Sayısal kuralları (50 puan vb.) kesin uygula.

    Ek Bilgiler:
    - 5 kez derse geç kalma = yarım gün devamsızlık.
    - Teşekkür: 70+, Takdir: 85+.
    - 5 gün özürsüz devamsızlık yapan belge alamaz.
    - Ortalama 50 altı ise sınıf geçilemez.

    Bağlam:
    {baglam}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": soru}
            ],
            model="llama-3.1-8b-instant",
            temperature=0
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Hata: {str(e)}"

# --- 4. SOHBET GEÇMİŞİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        cevap = okul_asistani_sorgula(prompt)
        st.write(cevap)
        st.session_state.messages.append({"role": "assistant", "content": cevap})
