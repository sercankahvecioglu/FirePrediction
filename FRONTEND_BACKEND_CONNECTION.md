# Frontend-Backend Connection Setup

## Conexión Completa Frontend-Backend

Este documento describe cómo he conectado el frontend React con el backend FastAPI para que funcione perfectamente.

## Cambios Realizados

### 1. Configuración del Proxy en Vite (client/vite.config.ts)
```typescript
server: {
  host: "::",
  port: 8080,
  proxy: {
    '/api': {
      target: 'http://localhost:5001',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, ''),
    },
  },
}
```
- **Propósito**: Redirige todas las peticiones de `/api/*` al backend en puerto 5001
- **Rewrite**: Elimina el prefijo `/api` para que coincida con las rutas del backend

### 2. Actualización del API Client (client/src/services/api.ts)
- **Flujo Asíncrono**: Implementé el patrón job-based que usa el backend
- **Polling**: El frontend hace polling cada 2 segundos para verificar el estado del job
- **Progress Tracking**: Muestra progreso en tiempo real (0-100%)
- **Error Handling**: Manejo robusto de errores

Flujo de trabajo:
1. `submitImageForCloudDetection()` - Envía imagen
2. `pollJobStatus()` - Monitorea progreso 
3. `getProcessingResult()` - Obtiene resultado final

### 3. Componente ImageUploader Mejorado
- **Barra de Progreso**: Muestra progreso visual con porcentaje
- **Mensajes de Estado**: Muestra mensajes del backend en tiempo real
- **Props Adicionales**: `progress` y `progressMessage`

### 4. Backend Enhancements (server/app.py)

#### CORS y Static Files:
```python
# CORS para permitir frontend
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"])

# Servir imágenes procesadas
app.mount("/static/images", StaticFiles(directory=DISPLAY_FOLDER), name="images")
```

#### Generación de Imágenes:
- **RGB Visualization**: Usa las primeras 3 bandas para crear imagen RGB
- **Cloud Mask**: Visualización de máscara de nubes basada en metadata
- **Placeholders**: Imágenes placeholder para forest y fire (no implementados aún)

#### URLs Corregidas:
```python
# Antes: rutas de archivos locales
rgb_image_url=os.path.join(DISPLAY_FOLDER, f"{job_id}_rgb.png")

# Después: URLs HTTP válidas  
rgb_image_url=f"/static/images/{job_id}_rgb.png"
```

## Cómo Probar la Conexión

### 1. Iniciar el Backend
```bash
cd server
python app.py
```
El servidor debería ejecutarse en `http://localhost:5001`

### 2. Verificar Backend (Opcional)
```bash
# Desde la raíz del proyecto
python test_connection.py
```

### 3. Iniciar el Frontend
```bash
cd client
npm run dev
# o si usas bun:
bun run dev
```
El frontend debería ejecutarse en `http://localhost:8080`

### 4. Probar Upload
1. Ve a `http://localhost:8080`
2. Sube un archivo `.npy`
3. Deberías ver:
   - Barra de progreso
   - Mensajes de estado en tiempo real
   - Imagen RGB cuando complete
   - Máscaras de cloud, forest y fire al final

## Arquitectura de la Conexión

```
Frontend (Port 8080)
    ↓ /api/submit-image/cloud-detection
Vite Proxy (rewrite to /)
    ↓ /submit-image/cloud-detection
Backend (Port 5001)
    ↓ Job ID
Frontend polls /api/job-status/{id}
    ↓ Status updates
Backend processes image
    ↓ Generates PNGs in DISPLAY folder
Frontend gets /api/get-result/{id}
    ↓ URLs to /static/images/*.png
Frontend displays images
```

## Flujo de Datos

1. **Upload**: Frontend → Backend (`/submit-image/cloud-detection`)
2. **Job Creation**: Backend crea job asíncrono
3. **Polling**: Frontend consulta estado cada 2s (`/job-status/{id}`)
4. **Processing**: Backend procesa tiles y actualiza progreso
5. **Image Generation**: Backend crea PNGs en DISPLAY folder
6. **Completion**: Frontend obtiene URLs de imágenes (`/get-result/{id}`)
7. **Display**: Frontend muestra imágenes via `/static/images/`

## Características Implementadas

✅ **Conexión Frontend-Backend completa**
✅ **Proxy configurado correctamente**
✅ **Flujo asíncrono con jobs**
✅ **Progress tracking en tiempo real**
✅ **Generación de imágenes RGB y cloud mask**
✅ **Servir archivos estáticos**
✅ **CORS configurado**
✅ **Error handling robusto**
✅ **URLs HTTP válidas**

## Próximos Pasos (Opcional)

- [ ] Implementar forest detection real
- [ ] Implementar fire prediction real  
- [ ] Añadir más tipos de visualización
- [ ] Mejorar algoritmos de cloud detection
- [ ] Añadir autenticación
- [ ] Optimizar rendimiento

## Estructura de Archivos Modificados

```
client/
├── vite.config.ts (proxy config)
├── src/
│   ├── services/api.ts (job-based API)
│   ├── pages/Index.tsx (progress handling)
│   └── components/ImageUploader/ImageUploader.tsx (progress UI)

server/
├── app.py (CORS, static files, image generation)
└── data/DISPLAY/ (generated images)

test_connection.py (testing script)
```

La conexión está completamente funcional y lista para usar.
