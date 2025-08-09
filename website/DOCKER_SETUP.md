# 🐳 FlameSentinels Docker Kurulumu

Bu dokümanda FlameSentinels projesini Docker ile nasıl çalıştıracağınızı öğreneceksiniz.

## 📋 Gereksinimler

- Docker
- Docker Compose
- En az 4GB RAM
- 10GB boş disk alanı

## 🚀 Hızlı Başlangıç

### 1. Projeyi Klonlayın
```bash
git clone <repository-url>
cd flamesentinels
```

### 2. Docker Container'larını Başlatın
```bash
docker-compose up -d
```

### 3. Servisleri Kontrol Edin
```bash
docker-compose ps
```

## 🌐 Erişim Adresleri

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **MongoDB**: localhost:27018
- **Mongo Express**: http://localhost:8082 (admin/password)

## 📁 Container Yapısı

### Frontend Container
- **Port**: 3000
- **Teknoloji**: React + TypeScript
- **Dockerfile**: `client/Dockerfile`

### Backend Container
- **Port**: 5000
- **Teknolojiler**: Node.js + Python
- **AI Model**: `server/ai_model.py`
- **Dockerfile**: `server/Dockerfile`

### MongoDB Container
- **Port**: 27018
- **Veritabanı**: flamesentinels
- **Kullanıcı**: admin/password

## 🔧 Geliştirme Modu

### Sadece Frontend
```bash
cd client
npm install
npm start
```

### Sadece Backend
```bash
cd server
npm install
npm run dev
```

### Tüm Servisler (Docker ile)
```bash
docker-compose up -d
```

## 📊 AI Model Testi

### Python AI Modelini Test Et
```bash
# Backend container'ına bağlan
docker exec -it flamesentinels-backend bash

# Test dosyası oluştur
python3 -c "import numpy as np; np.save('test.npy', np.random.rand(100, 100))"

# AI modelini test et
python3 ai_model.py test.npy
```

### Dosya Yükleme Testi
1. Frontend'e git: http://localhost:3000
2. .npy veya .tif dosyası yükle
3. Analiz sonuçlarını kontrol et

## 🛠️ Sorun Giderme

### Container'lar Başlamıyor
```bash
# Logları kontrol et
docker-compose logs

# Container'ları yeniden başlat
docker-compose down
docker-compose up -d
```

### Port Çakışması
```bash
# Kullanılan portları kontrol et
lsof -i :3000
lsof -i :5000
lsof -i :27018

# docker-compose.yml'da portları değiştir
```

### Python Bağımlılıkları
```bash
# Backend container'ında Python paketlerini güncelle
docker exec -it flamesentinels-backend pip install -r requirements.txt
```

## 📝 API Endpoints

### Dosya Yükleme
```bash
POST /api/upload
Content-Type: multipart/form-data
Body: file (.npy, .tif)
```

### Analiz
```bash
POST /api/analyze
Content-Type: application/json
Body: { "fileId": "..." }
```

### Durum Kontrolü
```bash
GET /api/status
```

## 🔍 Logları İzleme

### Tüm Servisler
```bash
docker-compose logs -f
```

### Belirli Servis
```bash
docker-compose logs -f frontend
docker-compose logs -f backend
docker-compose logs -f mongodb
```

## 🧹 Temizlik

### Container'ları Durdur
```bash
docker-compose down
```

### Tüm Verileri Sil
```bash
docker-compose down -v
docker system prune -a
```

## 📈 Performans

### Resource Kullanımı
```bash
docker stats
```

### Container Boyutları
```bash
docker images
```

## 🔐 Güvenlik

- MongoDB şifresi: `password`
- Mongo Express: `admin/password`
- Production'da environment variables kullanın

## 📞 Destek

Sorun yaşarsanız:
1. Logları kontrol edin
2. Container durumlarını kontrol edin
3. Port çakışmalarını kontrol edin
4. Disk alanını kontrol edin 