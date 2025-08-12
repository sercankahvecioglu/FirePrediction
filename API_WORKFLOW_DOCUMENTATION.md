# 🛰️ Satellite Image Analysis API - Asynchronous Workflow

## 📋 Overview

The API has been restructured to follow an **asynchronous job-based processing schema** with three separate analysis workflows:

1. **☁️ Cloud Detection**
2. **🌲 Forest Detection** 
3. **🔥 Fire Prediction**

Each analysis follows the same 5-step workflow pattern:

## 🚀 API Endpoints by Step

### Step 1: 🖼️ Submit Image for Processing

| Analysis Type | Endpoint | Description |
|---------------|----------|-------------|
| ☁️ Cloud Detection | `POST /submit-image/cloud-detection` | Submit image for cloud mask generation |
| 🌲 Forest Detection | `POST /submit-image/forest-detection` | Submit image for forest area identification |
| 🔥 Fire Prediction | `POST /submit-image/fire-prediction` | Submit image for fire risk analysis |

**Request Format:**
```http
POST /submit-image/{analysis-type}
Content-Type: multipart/form-data

file: satellite_image.npy
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "message": "Job created successfully",
  "estimated_time": 120
}
```

### Step 2-3: 🔄 Check Processing Status

| Endpoint | Description |
|----------|-------------|
| `GET /job-status/{job_id}` | Monitor job progress and status |

**Response:**
```json
{
  "job_id": "uuid-string",
  "task": "cloud_detection",
  "status": "processing",
  "created_at": "2025-08-12T10:30:00",
  "completed_at": null,
  "progress": 75,
  "message": "Generating cloud mask..."
}
```

**Status Values:**
- `pending`: Job created, waiting to start
- `processing`: Currently being processed
- `completed`: ✅ Ready for result retrieval
- `failed`: ❌ Processing error occurred

### Step 4: 📊 Get Processing Results

| Endpoint | Description |
|----------|-------------|
| `GET /get-result/{job_id}` | Retrieve analysis results and metadata |

**Response:**
```json
{
  "job_id": "uuid-string",
  "task": "fire_prediction",
  "image_url": "/download-image/{job_id}",
  "metadata": {
    "risk_levels": {
      "low": 35.2,
      "medium": 41.3,
      "high": 18.1,
      "critical": 5.4
    },
    "average_risk_score": 0.68,
    "confidence_score": 0.87,
    "processing_time": 284.5
  },
  "processing_time": 284.5
}
```

### Step 5: 📥 Download Processed Image

| Endpoint | Description |
|----------|-------------|
| `GET /download-image/{job_id}` | Download the final processed image |

**Response:** Binary image file (PNG/TIFF format)

## 🎯 Analysis Types and Outputs

### ☁️ Cloud Detection Analysis

**Purpose:** Identify and mask cloud coverage in satellite imagery

**Input Requirements:**
- Multi-spectral satellite image (RGB + NIR + SWIR bands recommended)
- Supported formats: `.npy`, `.tiff`, `.tif`

**Output Products:**
- Binary cloud mask overlay
- Cloud coverage percentage statistics
- Clear area visualization

**Processing Time:** ~2 minutes

**Implementation Status:** 🚧 **SKELETON FUNCTION** - Needs implementation

---

### 🌲 Forest Detection Analysis  

**Purpose:** Identify forest areas and vegetation types

**Input Requirements:**
- Multi-spectral satellite image with vegetation bands
- NDVI/EVI calculation capability preferred

**Output Products:**
- Forest boundary delineation
- Forest type classification (deciduous/coniferous/mixed)
- Vegetation health assessment
- NDVI/EVI index maps

**Processing Time:** ~3 minutes

**Implementation Status:** 🚧 **SKELETON FUNCTION** - Needs implementation

---

### 🔥 Fire Risk Prediction Analysis

**Purpose:** Predict fire risk probability and generate early warnings

**Input Requirements:**
- Multi-spectral satellite image with thermal bands
- Environmental data integration (weather, elevation)

**Output Products:**
- Fire risk probability heatmap
- Risk level classifications (Low/Medium/High/Critical)
- Vulnerable area identification
- 📧 **Automatic email alerts** for high-risk zones

**Processing Time:** ~5 minutes

**Alert System:**
- Triggers when >20% of area has high risk OR average risk >0.6
- Sends email notifications to configured recipients
- Integration with emergency response systems

**Implementation Status:** 🚧 **SKELETON FUNCTION** - Needs implementation

## 🔧 Job Management Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | GET | List all processing jobs |
| `/job/{job_id}` | DELETE | Delete job and cleanup files |
| `/` | GET | API health check and info |

## 📝 Implementation Notes

### Current Status
All three analysis functions are implemented as **skeleton functions** with:
- ✅ Complete API structure and routing
- ✅ Job management and status tracking
- ✅ Asynchronous processing workflow
- ✅ Email alert system (fire prediction)
- 🚧 **TODO:** Actual algorithm implementation

### Integration Points
The skeleton functions include detailed TODO comments indicating where to integrate:

1. **Cloud Detection:** `utils/cloud_detection/clouddetector.py`
2. **Forest Detection:** `utils/forest_detection/` modules
3. **Fire Prediction:** `utils/fire_prediction/unet.py` and `train_test.py`

### Example Client Usage

```python
import requests
import time

# Step 1: Submit image
with open('satellite_image.npy', 'rb') as f:
    response = requests.post(
        'http://localhost:5001/submit-image/fire-prediction',
        files={'file': f}
    )
job_data = response.json()
job_id = job_data['job_id']

# Step 2-3: Monitor progress
while True:
    status = requests.get(f'http://localhost:5001/job-status/{job_id}').json()
    print(f"Progress: {status['progress']}% - {status['message']}")
    
    if status['status'] == 'completed':
        break
    elif status['status'] == 'failed':
        print("Processing failed!")
        break
    
    time.sleep(5)  # Poll every 5 seconds

# Step 4: Get results
result = requests.get(f'http://localhost:5001/get-result/{job_id}').json()
print(f"Analysis complete! Risk score: {result['metadata']['average_risk_score']}")

# Step 5: Download image
image_response = requests.get(f'http://localhost:5001/download-image/{job_id}')
with open(f'fire_risk_heatmap_{job_id}.png', 'wb') as f:
    f.write(image_response.content)
```

## 🚨 Alert System (Fire Prediction Only)

The fire prediction analysis includes an automatic alert system:

**Trigger Conditions:**
- High-risk areas >20% of total area
- Average risk score >0.6

**Alert Actions:**
- 📧 Email notifications sent via `utils/mail/send_mail.py`
- Alert metadata included in processing results
- Configurable recipient lists

## 🔄 Migration from Previous Version

The API has been completely restructured from a tile-based synchronous approach to an asynchronous job-based system:

**Previous (v1.0):** 
- Single endpoint with tile processing
- Synchronous operations
- Complex multi-step manual workflow

**Current (v2.0):**
- Separate endpoints for each analysis type
- Asynchronous job-based processing
- Simple 5-step workflow
- Better scalability and user experience

---

## 🎉 Ready for Implementation!

The API structure is complete and ready for algorithm implementation. Each analysis function contains detailed documentation about expected inputs, outputs, and integration points with existing utility modules.
