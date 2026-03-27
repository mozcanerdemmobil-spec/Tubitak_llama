import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="MEB Asistanı", page_icon="🎓")
st.title("🎓 MEB Ortaöğretim Yönetmelik Asistanı")

# --- 2. VERİ TABANI VE MODEL HAZIRLIĞI ---
@st.cache_resource # Veri tabanını her seferinde tekrar yüklememesi için önbelleğe alıyoruz
def load_data():
    # Colab'de kullandığın embedding modelinin aynısı olmalı (Genelde budur)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Veri tabanını klasörden yüklüyoruz
    vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)
    return vector_db

# Sidebar - API Key
with st.sidebar:
    api_key = st.text_input("Groq API Key", type="password")
    if not api_key:
        st.warning("Devam etmek için API Key giriniz.")
        st.stop()

client = Groq(api_key=api_key)
vector_db = load_data()

# --- 3. SORGULAMA FONKSİYONU ---
def okul_asistani_sorgula(soru):
    # Benzerlik araması
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # SENİN O EFSANE SİSTEM MESAJIN (Kısaltmadan buraya ekliyoruz)
    system_prompt = """Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzman, teknik ve resmi bir asistansın.
    Kritik Kurallar:
    1. SADECE sana verilen 'Bağlam' içindeki bilgileri kullan.
    ... (Buraya senin koddaki 22 kuralın tamamını ve Ek Bilgileri aynen yapıştır) ...
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=1000
    )
    return chat_completion.choices[0].message.content

# --- 4. ARAYÜZ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yönetmelik taranıyor..."):
            cevap = okul_asistani_sorgula(prompt)
            st.write(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
