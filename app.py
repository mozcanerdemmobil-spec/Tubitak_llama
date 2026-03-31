import streamlit as st
import requests
import base64

from groq import Groq

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings


# --- 1. SAYFA AYARLARI ---

st.set_page_config(page_title="MEB Asistanı", page_icon="🎓")

st.title("🎓 MEB Ortaöğretim Yönetmelik Asistanı")


# Buton ve Gösterim Mantığı
if "show_pdf" not in st.session_state:
    st.session_state.show_pdf = False

col1, col2 = st.columns([1, 5])

with col1:
    if st.button("📄 Programı Göster"):
        st.session_state.show_pdf = not st.session_state.show_pdf

# Eğer butona basıldıysa PDF'i göster
if st.session_state.show_pdf:
    st.info("PDF aşağıda görüntüleniyor. Kapatmak için tekrar butona basabilirsiniz.")
    display_pdf(pdf_url)
else:
    st.write("PDF'i görmek için butona tıklayın.")


# --- 2. VERİ TABANI VE MODEL HAZIRLIĞI ---

@st.cache_resource

def load_data():

    # Embedding modelini yüklüyoruz

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Veri tabanını klasörden yüklüyoruz (Klasör adının doğru olduğundan emin ol)

    vector_db = Chroma(persist_directory="./okul_asistani_db", embedding_function=embeddings)

    return vector_db



# Sidebar - API Key Girişi

with st.sidebar:

    st.header("Ayarlar")

    api_key = st.text_input("Groq API Key", type="password").strip()

    if not api_key:

        st.warning("Lütfen devam etmek için API Key giriniz.")

        st.stop()

# Combo box (selectbox) oluşturma

secenekler = ["Görevli", "Öğrenci 9-A/BL", "Öğrenci 9-B/BL", "Öğrenci 9-C/BL", "Öğrenci 9-D/BL", "Öğrenci 9-E/EL", "Öğrenci 9-F/EL", "Öğrenci 9-G/EL", "Öğrenci 9-H/EL", "Öğrenci 9-I/EL", "Öğrenci 9/ATP", "Öğrenci 10A/BL", "Öğrenci 10B/BL", "Öğrenci 10C/EN", "Öğrenci 10D/HB", "Öğrenci 10E/HB", "Öğrenci 10 ATP", "Öğrenci 11A/BL", "Öğrenci 11B/BL", "Öğrenci 11C/EN", "Öğrenci 11D/HB", "Öğrenci 11 ATP", "Öğrenci 12A/BL", "Öğrenci 12B/BL", "Öğrenci 12F/HB", "Öğrenci 12G/EN", "Öğrenci 12 ATP"]

secilen_rol = st.sidebar.selectbox(

    "Lütfen rolünüzü seçiniz:",

    secenekler

)



# Seçilen değere göre ana ekranda işlem yapma

st.write(f"Şu anki profil: **{secilen_rol}**")



# Gerekli nesneleri oluşturuyoruz

client = Groq(api_key=api_key)

vector_db = load_data()



# --- 3. SORGULAMA FONKSİYONU ---

def okul_asistani_sorgula(soru):

    # Benzerlik araması yaparak bağlamı çekiyoruz

    docs = vector_db.similarity_search(soru, k=5)

    baglam = "\n\n".join([doc.page_content for doc in docs])



    # SENİN KRİTİK KURALLARININ TAMAMI

    system_prompt = f"""Sen MEB Ortaöğretim Kurumları Yönetmeliği konusunda uzman, teknik ve resmi bir asistansın.



    Kritik Kurallar:

    1. SADECE sana verilen 'Bağlam' içindeki bilgileri kullan.

    2. Eğer cevap bağlamda yoksa, 'Bu konuyla ilgili yönetmelikte net bir bilgi bulamadım' de. ASLA dış dünyadan bildiğin bilgileri ekleme (halüsinasyonu önler).

    3. Cevaplarını maddeler halinde ve resmi bir dille ver.

    4. Kişisel yorum yapma, veli veya okul müdürü gibi rolleri sınav katılımcılarıyla karıştırma.

    5. Cevabını aşağıdaki formatta ver:

    Cevap:

    - (Kısa ve net cevap)

    6. Birincil kaynağın her zaman Bağlamdır, ikincil kaynağın ise Yönetmelik temelli yorumlanmış Ek bilgilerdir, bunun dışında hiç bir yerden bilgi eklemesi yapma, Eğer Bağlam ile İkincil kaynak çelişkisi varsa Bağlam doğrudur, Bağlam ve İkincil kaynak aynı anda kullanılamaz, İlk bakacak olduğun yer her zaman Bağlamdır eğer Bağlamda konu hakkında bilgi yok ise İkincil Kaynağa geç eğer ordada bilgi yok ise "Konu Hakkında bir Bilgi bulamadım" yaz.

    7. Aynı cevap içinde ÇELİŞKİ OLAMAZ.

      - Bir bilgi verildiğinde, onu çürüten ikinci bir ifade yazma.

    8. "Ek Bilgi" başlığı KULLANMA.

      - Ek bilgiler, normal cevabın içine yedirilerek yazılmalıdır.

    9. Eğer Ek Bilgi kullanıldıysa:

      - "yönetmelikte net bilgi yok" ifadesi KULLANMA.

      - Bu iki durum aynı cevapta birlikte bulunamaz.

    10. Eğer cevap Ek Bilgiden geliyorsa:

      - Cümleler kesin ama sade olmalı.

      - Örneğin: "Bu durumda sınıfı geçemezsiniz."

    11. SAYISAL sorularda (50, 70, 85 gibi):

      - Kesin cevap ver.

      - Yuvarlak veya kaçamak cevap yazma.

    12. Eğer kullanıcı sorusu:

      - “geçer miyim”

      - “alabilir miyim”

      - “engel olur mu”

      ise:

      → Cevap mutlaka “Evet” veya “Hayır” ile başlamalıdır.

    13. Aynı cevapta hem:

      - hüküm verip

      - hem de belirsizlik belirtme

      YASAKTIR.

      Örnek yasak:

      ❌ "Geçemezsiniz. Ancak yönetmelikte net bilgi yoktur."

    14. Cevap formatı sadece şu olmalı:

    Cevap:

    - ...



    Ek Bilgiler (YÖNETMELİK TEMELLİ YORUMLANMIŞ - İKİNCİL KAYNAK):

    1. Okula geç kalan öğrenci yarım gün devamsız sayılmaz.

      - Geç kalma, devamsızlık değil “derse geç girme” olarak değerlendirilir.

      - Ancak 5 kez derse geç kalma, sistemde yarım gün devamsızlığa dönüştürülür.

    2. Destekleme ve Yetiştirme Kursları (DYK) için devam zorunluluğu vardır.

      - Mazeretsiz devamsızlık süresi, toplam ders saatinin %20’sini (1/5) aşarsa kurs kaydı silinir.

      - Ancak bu devamsızlık örgün eğitimdeki (hafta içi) devamsızlığı etkilemez.

    3. Teşekkür belgesi alabilmek için dönem sonu başarı puanı en az 70 olmalıdır.

    4. Takdir belgesi alabilmek için dönem sonu başarı puanı en az 85 olmalıdır.

    5. Belge (Takdir / Teşekkür) almak için özürsüz devamsızlık süresi 5 günü geçmemelidir.

      - 5 gün ve üzeri özürsüz devamsızlık yapan öğrenci belge alamaz.

    6. Yıl sonu başarı puanı 50 ve üzeri olan öğrenciler, başarısız dersleri olsa bile sorumlu olarak sınıf geçebilir.

    7. Yıl sonu başarı puanı 50’nin altında olan öğrenciler sınıfı doğrudan geçemez.

    8. Bir dersten başarısız olmak (zayıf almak), ortalama 50’nin altındaysa sınıf geçmeye engel olur.

    9. Ortalama 50’nin altında olan öğrenciler, zayıf sayısına bakılmaksızın başarısız sayılır.

    10. Sorumluluk sınavına, başarısız dersi bulunan öğrenciler girer.

        - Bu sınavlar başarısız olunan derslerden yapılır.

    11. Ortak (Türk Dili ve Edebiyatı gibi) derslerden başarısız olan öğrenciler, ortalamaları yeterli olsa bile sorumlu olarak sınıf geçer.

    12. Disiplin cezası alan öğrenciler, davranış puanına bağlı olarak Onur Belgesi alamayabilir.

    13. Okul birinciliği belirlenirken yalnızca akademik başarı değil, disiplin durumu ve genel davranışlar da dikkate alınır.

    14. İşletmede staj yapan öğrencinin kaza geçirmesi durumunda:

        - İşletme durumu derhal okula bildirir

        - Gerekli resmi kayıtlar tutulur

        - Süreç okul ve işletme koordinasyonunda yürütülür

    15. "Ek Bilgi" başlığı sadece gerekli olduğunda kullanılmalıdır.

    16. "Ek Bilgi" içeriği, ana cevabı DESTEKLEMELİDİR.

    17. "Ek Bilgi" kısmına SADECE soruyla doğrudan ilgili bilgiler yazılır.

    18. Aynı cevap içinde çelişki bulunamaz.

    19. Eğer cevap "Evet" veya "Hayır" gerektiriyorsa cevap mutlaka "Evet" veya "Hayır" ile başlamalıdır.

    20. Sayısal kurallar kesin uygulanmalıdır: Puan 50 altıysa geçemez, 50 ve üzeriyse geçebilir.

    21. Ek Bilgi sadece soruya doğrudan katkı sağlıyorsa yazılır.

    22. Hem Bağlamda hem Ek Bilgilerde bilgi yoksa: "Bu konuyla ilgili yönetmelikte net bir bilgi bulamadım" yazılır.



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

        return f"Hata: {str(e)}"



# --- 4. ARAYÜZ (SOHBET GEÇMİŞİ) ---

if "messages" not in st.session_state:

    st.session_state.messages = []



# Eski mesajları ekrana bas

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])



# Kullanıcı yeni bir şey yazarsa

if prompt := st.chat_input("Sorunuzu buraya yazın..."):

    # Kullanıcı mesajını kaydet ve göster

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):

        st.write(prompt)



    # Cevap üret ve göster

    with st.chat_message("assistant"):

        with st.spinner("Yönetmelik taranıyor..."):

            cevap = okul_asistani_sorgula(prompt)

            st.write(cevap)

            st.session_state.messages.append({"role": "assistant", "content": cevap})



st.caption("⚠️ Asistan hata yapabilir. Verilen bilgilerin doğruluğunu her zaman resmi yönetmeliklerden kontrol edin. Ve eğer mümkünse sorularınızın tamamını bir kerede sorun.")
