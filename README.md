# Q-Learning ile Akıllı Ofis Klima Kontrolü  
**Elif Yılmaz – 25435004004**  

---

### Proje Özeti  
Bu proje, bir ofis binasında klima sistemini **insan konforunu korurken enerji tüketimini en aza indiren** bir yapay zekâ ajanı geliştirmek amacıyla hazırlanmıştır.  

Geliştirilen Q-Learning ajanı:  
- Çalışma saatlerinde (09:00–17:00) iç sıcaklığı **22–24 °C** konfor bandında tutar  
- Gece ve akşam klimayı **hiç açmaz**  
- Sabah 06:00’da “ön soğutma” yaparak mesai başlangıcında ideal sıcaklık sağlar  
- En sıcak öğlen saatlerinde “serbest salınım” (coasting) ile enerji tasarrufu yapar  

**Sonuç:**  
**%94.4** çalışma saati konfor başarısı  
**6.7 kW** günlük enerji tüketimi (rastgele ajan: 22.5 kW)  
Ortalama ödül: **+995** (rastgele ajan: –4814)  

---

### 1. Giriş ve Amaç  
Binalar küresel enerji tüketiminin yaklaşık %40’ından sorumludur ve bu tüketimin büyük kısmı HVAC sistemlerinden kaynaklanmaktadır. Geleneksel termostat tabanlı kontrol sistemleri yalnızca anlık sıcaklığa tepki verdiğinden hem konfor kaybına hem de yüksek enerji israfına yol açmaktadır.  

Bu projede, dış sıcaklık ve günün zaman dilimini dikkate alarak **proaktif** kararlar verebilen, enerji verimli ve konfor odaklı bir reinforcement learning ajanı tasarlanmıştır.

---

### 2. Simülasyon Ortamı: SmartOfficeEnv  

#### 2.1. Durum Uzayı  
- İç ortam sıcaklığı: 13 seviye (15–30 °C)  
- Dış ortam sıcaklığı: 11 seviye (−5–40 °C)  
- Zaman dilimi: 4 kategori (Gece, Sabah Hazırlık, Çalışma, Akşam)  

#### 2.2. Aksiyon Uzayı  
7 ayrık aksiyon:  
0 → Kapalı  1–3 → Soğutma (düşük/orta/yüksek)  4–6 → Isıtma  

#### 2.3. Fizik Modeli  
Her adımda iç sıcaklık şu denklemle güncellenir:  
$$T_{t+1} = T_t + 0.08 \cdot (T_{\text{dış}} - T_t) + \Delta T_{\text{aksiyon}}$$

#### 2.4. Ödül Fonksiyonu  
- Konfor bandında + klima kapalı → **+100**  
- Öğlen kalkan koruması ihlali (>22.8 °C’de kapatma) → **–500**  
- Gece/akşam aktif kullanım → **–200**  
- Konfor dışı + kapatma → ekstra **–300**

---

### 3. Uygulanan Algoritma  
- **Q-Learning** (tablo tabanlı)  
- $\alpha = 0.1$, $\gamma = 0.99$  
- Epsilon: 1.0 → 0.01 (decay = 0.9999)  
- Eğitim: **100.000 episod** (~1.2 dakika)

---

### 4. Deney Sonuçları  

| Metrik                           | Rastgele Ajan     | Eğitilmiş Q-Learning Ajan |
|----------------------------------|-------------------|----------------------------|
| Çalışma saati konfor oranı       | %11.1             | **%94.4**                  |
| Günlük enerji tüketimi           | 22.5 kW           | **6.7 kW**                 |
| Ortalama toplam ödül             | –4814             | **+995**                   |

#### Geliştirilen İleri Düzey Stratejiler  
- Sabah 06:00’da “ön soğutma” yaparak mesai başlangıcında ideal sıcaklık sağlar  
- En sıcak öğlen saatlerinde “serbest salınım” (coasting) ile enerji tasarrufu yapar  
- Gece ve akşam klimayı hiç açmaz  
- Konfor sınırı aşıldığında hızlı müdahale eder  

---

### 5. Dosya Yapısı  
- `Smart_Office_AC.ipynb` → Tüm kod, eğitim ve analiz  
- `best_q_table.npy` → En iyi Q-tablosu (otomatik kaydedilir)  
- `smart_office_result.gif` → 24 saatlik performans animasyonu  
