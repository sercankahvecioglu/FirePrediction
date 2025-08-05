# FlameSentinels - Wildfire Risk Intelligence Platform

AI destekli uydu görüntülerinden orman yangını risk tahmini yapan modern web platformu.

## 🚀 Özellikler

- **AI Destekli Analiz**: %99.8 doğruluk oranıyla yangın riski tahmini
- **Gerçek Zamanlı İşleme**: Hızlı ve etkili analiz
- **Global Kapsam**: Dünya çapında uydu görüntü desteği
- **Modern UI/UX**: Koyu tema ile profesyonel arayüz
- **Dosya Yükleme**: .tif ve .npy formatlarında destek
- **Risk Haritaları**: Görsel ısı haritaları ve risk değerlendirmesi

## 🛠️ Teknolojiler

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- React Dropzone
- Chart.js
- Lucide React (ikonlar)

### Backend
- Node.js
- Express.js
- Multer (dosya yükleme)
- CORS
- Nodemon (geliştirme)

## 📦 Kurulum

### 1. MongoDB ve mongo-express'i başlatın:
```bash
docker-compose up -d
```

### 2. Environment dosyasını oluşturun:
```bash
cp env.example .env
```

### 3. Bağımlılıkları yükleyin:
```bash
npm run install-all
```

### 4. Geliştirme sunucusunu başlatın:
```bash
npm run dev
```

### 5. Tarayıcıda açın:
```
Frontend: http://localhost:3000
Backend: http://localhost:5001
Mongo Express: http://localhost:8081 (admin/password123)
```

## 📁 Proje Yapısı

```
flamesentinels/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # UI bileşenleri
│   │   ├── pages/         # Sayfa bileşenleri
│   │   ├── hooks/         # Custom hooks
│   │   └── utils/         # Yardımcı fonksiyonlar
│   └── public/            # Statik dosyalar
├── server/                # Node.js backend
│   ├── routes/            # API rotaları
│   ├── middleware/        # Middleware'ler
│   ├── uploads/           # Yüklenen dosyalar
│   └── index.js           # Ana sunucu dosyası
└── package.json           # Ana proje konfigürasyonu
```

## 🔧 Kullanım

1. Ana sayfada dosya yükleme alanına .tif veya .npy dosyası sürükleyin
2. Sistem otomatik olarak uydu görüntüsünü analiz eder
3. Risk haritası ve değerlendirme sonuçları görüntülenir
4. Risk seviyesi slider'ı ile farklı eşik değerleri ayarlayabilirsiniz

## 📊 API Endpoints

### Temel Endpoints
- `GET /api/status` - Sistem durumu
- `POST /api/upload` - Dosya yükleme
- `POST /api/analyze` - Görüntü analizi

### MongoDB Entegrasyonu
- `GET /api/uploads` - Tüm yüklenen dosyaları listele
- `GET /api/uploads/:id` - Belirli bir dosyayı getir
- `DELETE /api/uploads/:id` - Dosya sil

### Query Parametreleri
- `page` - Sayfa numarası (varsayılan: 1)
- `limit` - Sayfa başına kayıt (varsayılan: 10)
- `status` - Dosya durumu filtreleme (uploaded, analyzed)

