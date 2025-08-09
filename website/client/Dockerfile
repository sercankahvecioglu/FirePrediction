FROM node:18-alpine

WORKDIR /app

# Package.json ve package-lock.json dosyalarını kopyala
COPY package*.json ./

# Bağımlılıkları yükle
RUN npm install

# Kaynak kodları kopyala
COPY . .

EXPOSE 3000

# Development modunda başlat
CMD ["npm", "start"] 