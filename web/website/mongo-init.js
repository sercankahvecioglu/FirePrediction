// MongoDB başlatma script'i
db = db.getSiblingDB('flamesentinels');

// Uploads koleksiyonu için index'ler oluştur
db.createCollection('uploads');
db.uploads.createIndex({ "uploadTime": -1 });
db.uploads.createIndex({ "filename": 1 });
db.uploads.createIndex({ "status": 1 });

print('✅ FlameSentinels veritabanı ve koleksiyonları oluşturuldu');
print('📊 Veritabanı: flamesentinels');
print('📁 Koleksiyon: uploads'); 