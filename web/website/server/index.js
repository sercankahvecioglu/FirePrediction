const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { MongoClient, ObjectId } = require('mongodb');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 5001;

// MongoDB bağlantı konfigürasyonu
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017';
const DB_NAME = 'flamesentinels';
const COLLECTION_NAME = 'uploads';

let db;
let uploadsCollection;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Uploads klasörünü oluştur
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Multer konfigürasyonu
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['.tif', '.npy'];
    const ext = path.extname(file.originalname).toLowerCase();
    if (allowedTypes.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error('Sadece .tif ve .npy dosyaları kabul edilir!'), false);
    }
  },
  limits: {
    fileSize: 100 * 1024 * 1024 // 100MB limit
  }
});

// MongoDB bağlantısını başlat
async function connectToMongoDB() {
  try {
    const client = new MongoClient(MONGODB_URI);
    await client.connect();
    console.log('✅ MongoDB\'ye başarıyla bağlandı');
    
    db = client.db(DB_NAME);
    uploadsCollection = db.collection(COLLECTION_NAME);
    
    // Index oluştur
    await uploadsCollection.createIndex({ uploadTime: -1 });
    await uploadsCollection.createIndex({ filename: 1 });
    
    console.log('✅ MongoDB koleksiyonları hazır');
  } catch (error) {
    console.error('❌ MongoDB bağlantı hatası:', error);
    process.exit(1);
  }
}

// Routes
app.get('/api/status', (req, res) => {
  res.json({
    status: 'online',
    message: 'FlameSentinels API çalışıyor',
    timestamp: new Date().toISOString(),
    mongodb: db ? 'connected' : 'disconnected'
  });
});

// Dosya yükleme endpoint'i
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Dosya yüklenmedi' });
    }

    const fileInfo = {
      _id: new ObjectId(),
      filename: req.file.filename,
      originalName: req.file.originalname,
      size: req.file.size,
      mimetype: req.file.mimetype,
      filePath: req.file.path,
      uploadTime: new Date(),
      status: 'uploaded'
    };

    // MongoDB'ye kaydet
    await uploadsCollection.insertOne(fileInfo);

    res.json({
      success: true,
      message: 'Dosya başarıyla yüklendi ve veritabanına kaydedildi',
      file: {
        id: fileInfo._id,
        filename: fileInfo.filename,
        originalName: fileInfo.originalName,
        size: fileInfo.size,
        uploadTime: fileInfo.uploadTime
      }
    });
  } catch (error) {
    console.error('Dosya yükleme hatası:', error);
    res.status(500).json({ error: 'Dosya yükleme hatası: ' + error.message });
  }
});

// Analiz endpoint'i (simüle edilmiş)
app.post('/api/analyze', async (req, res) => {
  try {
    const { filename, fileId } = req.body;
    
    if (!filename && !fileId) {
      return res.status(400).json({ error: 'Dosya adı veya ID gerekli' });
    }

    // MongoDB'den dosya bilgilerini al
    let fileDoc;
    if (fileId) {
      fileDoc = await uploadsCollection.findOne({ _id: new ObjectId(fileId) });
    } else {
      fileDoc = await uploadsCollection.findOne({ filename: filename });
    }

    if (!fileDoc) {
      return res.status(404).json({ error: 'Dosya bulunamadı' });
    }

    // Simüle edilmiş analiz sonuçları
    const analysisResult = {
      success: true,
      message: 'Analiz tamamlandı',
      fileId: fileDoc._id,
      results: {
        coordinates: {
          latitude: 34.0522,
          longitude: -118.2437
        },
        timestamp: new Date().toISOString(),
        resolution: '30m/pixel',
        riskAssessment: {
          highRisk: 12,
          mediumRisk: 35,
          lowRisk: 53,
          totalArea: 450
        },
        heatmapData: generateMockHeatmapData(),
        confidence: 99.8
      }
    };

    // Analiz sonucunu MongoDB'ye kaydet
    await uploadsCollection.updateOne(
      { _id: fileDoc._id },
      { 
        $set: { 
          status: 'analyzed',
          analysisResult: analysisResult.results,
          analyzedAt: new Date()
        }
      }
    );

    // Simüle edilmiş işlem süresi
    setTimeout(() => {
      res.json(analysisResult);
    }, 2000);

  } catch (error) {
    console.error('Analiz hatası:', error);
    res.status(500).json({ error: 'Analiz hatası: ' + error.message });
  }
});

// Tüm yüklenen dosyaları getir
app.get('/api/uploads', async (req, res) => {
  try {
    const { page = 1, limit = 10, status } = req.query;
    const skip = (page - 1) * limit;
    
    let query = {};
    if (status) {
      query.status = status;
    }
    
    const files = await uploadsCollection
      .find(query)
      .sort({ uploadTime: -1 })
      .skip(skip)
      .limit(parseInt(limit))
      .toArray();
    
    const total = await uploadsCollection.countDocuments(query);
    
    res.json({
      success: true,
      files: files.map(file => ({
        id: file._id,
        filename: file.filename,
        originalName: file.originalName,
        size: file.size,
        status: file.status,
        uploadTime: file.uploadTime,
        analyzedAt: file.analyzedAt
      })),
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit)
      }
    });
  } catch (error) {
    console.error('Dosyaları getirme hatası:', error);
    res.status(500).json({ error: 'Dosyaları getirme hatası: ' + error.message });
  }
});

// Belirli bir dosyayı getir
app.get('/api/uploads/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const file = await uploadsCollection.findOne({ _id: new ObjectId(id) });
    
    if (!file) {
      return res.status(404).json({ error: 'Dosya bulunamadı' });
    }
    
    res.json({
      success: true,
      file: {
        id: file._id,
        filename: file.filename,
        originalName: file.originalName,
        size: file.size,
        status: file.status,
        uploadTime: file.uploadTime,
        analyzedAt: file.analyzedAt,
        analysisResult: file.analysisResult
      }
    });
  } catch (error) {
    console.error('Dosya getirme hatası:', error);
    res.status(500).json({ error: 'Dosya getirme hatası: ' + error.message });
  }
});

// Dosya sil
app.delete('/api/uploads/:id', async (req, res) => {
  try {
    const { id } = req.params;
    
    const file = await uploadsCollection.findOne({ _id: new ObjectId(id) });
    
    if (!file) {
      return res.status(404).json({ error: 'Dosya bulunamadı' });
    }
    
    // Dosyayı fiziksel olarak sil
    if (fs.existsSync(file.filePath)) {
      fs.unlinkSync(file.filePath);
    }
    
    // MongoDB'den sil
    await uploadsCollection.deleteOne({ _id: new ObjectId(id) });
    
    res.json({
      success: true,
      message: 'Dosya başarıyla silindi'
    });
  } catch (error) {
    console.error('Dosya silme hatası:', error);
    res.status(500).json({ error: 'Dosya silme hatası: ' + error.message });
  }
});

// Mock heatmap verisi oluştur
function generateMockHeatmapData() {
  const data = [];
  for (let i = 0; i < 100; i++) {
    for (let j = 0; j < 100; j++) {
      data.push({
        x: i,
        y: j,
        risk: Math.random() * 100
      });
    }
  }
  return data;
}

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({ error: 'Dosya boyutu çok büyük (max: 100MB)' });
    }
  }
  res.status(500).json({ error: error.message });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint bulunamadı' });
});

// Sunucuyu başlat
async function startServer() {
  try {
    // MongoDB'ye bağlan
    await connectToMongoDB();
    
    // Sunucuyu başlat
    app.listen(PORT, () => {
      console.log(`🚀 FlameSentinels API sunucusu ${PORT} portunda çalışıyor`);
      console.log(`📡 API URL: http://localhost:${PORT}`);
      console.log(`🗄️  MongoDB: ${MONGODB_URI}`);
      console.log(`📊 Veritabanı: ${DB_NAME}`);
      console.log(`📁 Koleksiyon: ${COLLECTION_NAME}`);
    });
  } catch (error) {
    console.error('❌ Sunucu başlatma hatası:', error);
    process.exit(1);
  }
}

startServer(); 