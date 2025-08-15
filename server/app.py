import shutil
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import uvicorn
import uuid
import os
import json
from datetime import datetime
import asyncio
import sys
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import random
# Note: PIL/Pillow might not be installed, so we'll use matplotlib for all image generation

# ---- HOMEMADE LIBRARIES ---- #

# Add sys to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from utils.mail.send_mail import send_mail # Sending mail adverting higher fire risk

# Load from utils library
from utils import cloud_detection
from utils import fire_prediction  
from utils import preprocessing
from utils import forest_detection
from utils import data_api 
from utils import plotting

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Satellite Image Analysis API - Asynchronous Processing",
    description="API for processing satellite images with cloud detection, forest detection, and fire prediction using asynchronous job-based workflow",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response schemas
class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    """Status of image processing job"""
    job_id: str
    task: str  # 'cloud_detection', 'forest_detection', 'fire_prediction'
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime]
    tiles_to_process: int  # Number of tiles to process 
    tiles_processed: int  # Number of tiles processed
    successful_tiles: int  # Number of successfully processed tiles
    progress: int  # 0-100
    message: str

class ProcessedImageResponse(BaseModel):
    """Response model for processed image"""
    job_id: str
    task: str
    rgb_image_url: str
    cloud_image_url: str
    forest_image_url: str
    heatmap_image_url: str
    metadata: List[dict]
    processing_time: Optional[float] = None

# In-memory storage for demo purposes (use database in production)
processing_jobs = {}
processed_images = {}
forest_detection_job_id = {}
fire_prediction_job_id = {}

# ------ CONSTANTS ----- #

# allowed_formats = [".npy", ".tiff", ".tif"]
allowed_formats = [".npy"]

# Folders
DATA_PATH = os.path.join(os.getcwd(), "data")
RAW_IMAGES_PATH = os.path.join(DATA_PATH, "RAW_IMAGES")
CLOUD_IMAGES_PATH = os.path.join(DATA_PATH, "CLOUD_IMAGES")
TILES_IMAGES_PATH = os.path.join(DATA_PATH, "TILES_IMAGES")
FOREST_IMAGES_PATH = os.path.join(DATA_PATH, "FOREST_IMAGES")
FIRE_IMAGES_PATH = os.path.join(DATA_PATH, "FIRE_IMAGES")
METADATA_FOLDER = os.path.join(DATA_PATH, "METADATA")
DISPLAY_FOLDER = os.path.join(DATA_PATH, "DISPLAY")

# Create folders if necessary
os.makedirs(RAW_IMAGES_PATH, exist_ok=True)
os.makedirs(CLOUD_IMAGES_PATH, exist_ok=True)
os.makedirs(TILES_IMAGES_PATH, exist_ok=True)
os.makedirs(FOREST_IMAGES_PATH, exist_ok=True)
os.makedirs(FIRE_IMAGES_PATH, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)
os.makedirs(DISPLAY_FOLDER, exist_ok=True)

# Mount static files to serve processed images
app.mount("/static/images", StaticFiles(directory=DISPLAY_FOLDER), name="images")

# ---------------------- #

@app.get("/", tags=["Health Check"])
def healthcheck():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Welcome message and API status
    """
    return {
        "message": "Welcome to the Satellite Image Analysis API",
        "status": "operational",
        "version": "2.0.0",
        "available_analyses": ["cloud_detection", "forest_detection", "fire_prediction"]
    }

# ============================================================================
# STEP 1: SEND IMAGE - Submit image for processing
# ============================================================================

@app.post("/submit-image/cloud-detection", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_cloud_detection(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tile_size: int = 256
):
    """
    Submit a satellite image for cloud detection analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy)
    - Image should contain multi-spectral bands for optimal cloud detection
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for cloud detection
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())
    
    # Validate file format
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {allowed_formats}"
        )

    image = np.load(file.file)

    rows = math.ceil(image.shape[0] / tile_size)
    cols = math.ceil(image.shape[1] / tile_size)

    number_of_tiles = rows * cols

    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="cloud_detection",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        tiles_to_process=number_of_tiles,
        tiles_processed=0,
        successful_tiles=0,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(run_in_threadpool, process_cloud_detection, job_id, image, tile_size)

    # background_tasks.add_task(process_cloud_detection, job_id, image, tile_size)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Cloud detection job created successfully"
    )

@app.post("/submit-image/forest-detection", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_forest_detection(
    background_tasks: BackgroundTasks,
    cloud_job_id: str
):
    """
    Submit a satellite image for forest detection analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy, .tiff, .tif)
    - Image should contain vegetation indices (NDVI, EVI) for optimal forest detection
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for forest detection
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())

    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="forest_detection",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        tiles_to_process=processing_jobs[cloud_job_id].tiles_to_process,
        tiles_processed=0,
        successful_tiles=0,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(run_in_threadpool, process_forest_detection, job_id, cloud_job_id)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Forest detection job created successfully",
    )

@app.post("/submit-image/fire-prediction", response_model=JobResponse, tags=["Submit Image"])
async def submit_image_fire_prediction(
    background_tasks: BackgroundTasks,
    cloud_job_id: str
):
    """
    Submit a satellite image for fire risk prediction analysis.
    
    **Processing Flow:**
    1. Upload and validate satellite image
    2. Create asynchronous processing job
    3. Return job ID for tracking
    
    **Input:**
    - Satellite image file (supported formats: .npy, .tiff, .tif)
    - Image should contain multi-spectral bands including thermal for optimal fire prediction
    - Additional metadata (weather data, elevation) can improve predictions
    
    **Output:**
    - Job ID for tracking processing status
    - Estimated processing time
    - Automatic email alerts for high-risk areas
    
    **Next Steps:**
    1. Use job ID to check processing status with `/job-status/{job_id}`
    2. Once completed, retrieve result with `/get-result/{job_id}`
    
    Args:
        file (UploadFile): Satellite image file for fire risk prediction
        
    Returns:
        JobResponse: Job information including job_id for tracking progress
    """
    job_id = str(uuid.uuid4())
    
    # Create processing job
    processing_jobs[job_id] = JobStatus(
        job_id=job_id,
        task="fire_prediction",
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        tiles_to_process=processing_jobs[cloud_job_id].tiles_to_process,
        tiles_processed=0,
        successful_tiles=0,
        progress=0,
        message="Image uploaded successfully. Processing will start shortly."
    )
    
    # Start background processing
    background_tasks.add_task(process_fire_prediction, job_id, cloud_job_id)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Fire prediction job created successfully"
    )

# ============================================================================
# STEP 2 & 3: CHECK STATUS - Monitor processing progress
# ============================================================================

@app.get("/job-status/{job_id}", response_model=JobStatus, tags=["Check Status"])
def get_job_status(job_id: str):
    """
    Check the current status of a processing job.
    
    **Status Values:**
    - `pending`: Job created, waiting to start processing
    - `processing`: Job is currently being processed
    - `completed`: Job finished successfully, result available
    - `failed`: Job failed due to error
    
    **Progress Tracking:**
    - Progress percentage (0-100%)
    - Detailed status messages
    - Processing time information
    
    This endpoint should be polled regularly until status becomes `completed` or `failed`.
    
    Args:
        job_id (str): Unique identifier for the processing job
        
    Returns:
        JobStatus: Current job status and progress information
        
    Raises:
        HTTPException: If job_id is not found
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

# ============================================================================
# STEP 4: GET RESULT - Retrieve processed image
# ============================================================================

@app.get("/get-result/{job_id}", response_model=ProcessedImageResponse, tags=["Get Result"])
def get_processing_result(job_id: str):
    """
    Retrieve the result of a completed processing job.
    
    **Available when job status is `completed`**
    
    **Returns:**
    - URL to download processed image
    - Processing metadata and statistics
    - Analysis results and confidence scores
    
    **Result Types by Analysis:**
    
    **Cloud Detection:**
    - Binary cloud mask overlay
    - Cloud coverage percentage
    - Clear areas highlighted
    
    **Forest Detection:**
    - Forest area boundaries
    - Vegetation indices (NDVI, EVI)
    - Forest type classification
    
    **Fire Prediction:**
    - Fire risk heatmap
    - Risk level classifications
    - Vulnerable area identification
    - Automatic alerts for high-risk zones
    
    Args:
        job_id (str): Job identifier to get results for
        
    Returns:
        ProcessedImageResponse: Complete processing results including download URL
        
    Raises:
        HTTPException: If job not found or not completed
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if job_id not in processed_images:
        raise HTTPException(status_code=404, detail="Processed image not found")
    
    return processed_images[job_id]

# ============================================================================
# STEP 5: DOWNLOAD IMAGE - Get processed image file
# ============================================================================

@app.get("/download-image/{job_id}", tags=["Download Image"])
async def download_processed_image(job_id: str):
    """
    Download the processed image file.
    
    **File Formats:**
    - PNG: For visualization images (heatmaps, overlays)
    
    **File Naming Convention:**
    - `{task}_{job_id}_result.{ext}`
    - Example: `fire_prediction_abc123_result.png`
    
    Args:
        job_id (str): Job identifier to download image for
        
    Returns:
        FileResponse: The processed image file
        
    Raises:
        HTTPException: If job not found or file not available
    """

    print(f"Requesting images for job {job_id}")

    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Image not ready for download")
    
    # Get file path based on task type
    task = job.task

    if "cloud" in task:
        task_type = "cloud"
    elif "forest" in task:
        task_type = "forest"
    else:
        task_type = "fire"

    image_path = os.path.join(DISPLAY_FOLDER, f"{job_id}_{task_type}.png")
    # image_path = f"static/processed/{task}_{job_id}_result.png"
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Processed image file not found")
    
    return FileResponse(
        path=image_path,
        media_type="image/png",
        filename=f"{task}_{job_id}_result.png"
    )

@app.get("/download-picture/{name}", tags=["Download Image"])
async def download_picture(name: str):
    """
    Download a picture file by name.
    
    Args:
        name (str): Name of the picture file to download
        
    Returns:
        FileResponse: The picture file
        
    Raises:
        HTTPException: If file not found
    """
    # Validate filename to prevent path traversal
    if os.path.sep in name or os.path.altsep and os.path.altsep in name or name.startswith("..") or name.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(DISPLAY_FOLDER, name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Picture file not found")
    
    return FileResponse(
        path=file_path,
        media_type="image/png",
        filename=name
    )

# ============================================================================
# JOB MANAGEMENT ENDPOINTS
# ============================================================================

@app.get("/jobs", response_model=List[JobStatus], tags=["Job Management"])
def list_all_jobs():
    """
    List all processing jobs with their current status.
    
    Useful for monitoring and administration purposes.
    
    Returns:
        List[JobStatus]: List of all jobs with their status information
    """
    return list(processing_jobs.values())

@app.delete("/job/{job_id}", tags=["Job Management"])
def delete_job(job_id: str):
    """
    Delete a processing job and all associated data.
    
    This endpoint:
    1. Removes job from processing queue
    2. Deletes processed image files
    3. Cleans up temporary storage
    
    Args:
        job_id (str): Job identifier to delete
        
    Returns:
        dict: Deletion confirmation message
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from storage
    del processing_jobs[job_id]
    if job_id in processed_images:
        del processed_images[job_id]
    
    # TODO: Clean up files from disk
    
    return {"message": f"Job {job_id} deleted successfully"}

# ============================================================================
# FILE CLEANUP FUNCTIONS
# ============================================================================

@app.delete("/cleanup", tags=["File Management"])
def cleanup_files():
    """
    Clean up temporary files and folders used during processing.
    
    This endpoint:
    1. Deletes all files in the RAW_IMAGES_PATH
    2. Deletes all files in the CLOUD_IMAGES_PATH
    3. Deletes all files in the TILES_IMAGES_PATH
    4. Deletes all files in the FOREST_IMAGES_PATH
    5. Deletes all files in the FIRE_IMAGES_PATH
    6. Deletes all files in the METADATA_FOLDER
    7. Deletes all files in the DISPLAY_FOLDER

    Returns:
        dict: Cleanup confirmation message
    """
    try:
        shutil.rmtree(RAW_IMAGES_PATH)
        shutil.rmtree(CLOUD_IMAGES_PATH)
        shutil.rmtree(TILES_IMAGES_PATH)
        shutil.rmtree(FOREST_IMAGES_PATH)
        shutil.rmtree(FIRE_IMAGES_PATH)
        shutil.rmtree(METADATA_FOLDER)
        shutil.rmtree(DISPLAY_FOLDER)

        # Create folders if necessary
        os.makedirs(RAW_IMAGES_PATH, exist_ok=True)
        os.makedirs(CLOUD_IMAGES_PATH, exist_ok=True)
        os.makedirs(TILES_IMAGES_PATH, exist_ok=True)
        os.makedirs(FOREST_IMAGES_PATH, exist_ok=True)
        os.makedirs(FIRE_IMAGES_PATH, exist_ok=True)
        os.makedirs(METADATA_FOLDER, exist_ok=True)
        os.makedirs(DISPLAY_FOLDER, exist_ok=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning up files: {e}")

    return {"message": "Temporary files cleaned up successfully"}

# ============================================================================
# IMAGE VISUALIZATION FUNCTIONS
# ============================================================================

def create_cloud_mask_visualization(metadata: pd.DataFrame, image_shape: tuple, job_id: str, output_path: str):
    """
    Create cloud mask visualization from tile metadata.
    """
    try:
        # Create a binary mask based on cloud detection results
        cloud_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.float32)
        
        for _, row in metadata.iterrows():
            if pd.notna(row['cloud?']) and row['cloud?']:
                coords = row['tile_coordinates']
                if isinstance(coords, tuple) and len(coords) == 2:
                    i, j = coords
                    # Assuming tile size of 256x256 (you might need to adjust this)
                    tile_size = 256
                    end_i = min(i + tile_size, image_shape[0])
                    end_j = min(j + tile_size, image_shape[1])
                    cloud_mask[i:end_i, j:end_j] = row.get('cloud_percentage', 1.0)
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(cloud_mask, cmap='Blues', alpha=0.8)
        plt.colorbar(label='Cloud Coverage')
        plt.title(f'Cloud Detection Mask - {job_id}', fontsize=14, pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Cloud mask visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Error creating cloud mask visualization: {e}")
        # Create a placeholder image if cloud mask fails
        create_placeholder_image(output_path, "Cloud Mask\nNot Available")

def create_placeholder_image(output_path: str, text: str):
    """
    Create a placeholder image with text when actual processing fails.
    """
    plt.figure(figsize=(10, 10))
    plt.text(0.5, 0.5, text, fontsize=20, ha='center', va='center', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

def process_cloud_detection(job_id: str, image: np.ndarray, tiles_size: int):
    """
    Background task for cloud detection processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Apply cloud detection algorithms
    3. Generate cloud mask overlay
    4. Calculate cloud coverage statistics
    5. Save processed image and metadata
    
    **Algorithms Used:**
    - Threshold-based cloud detection
    - Machine learning cloud classification
    - Multi-spectral band analysis
    
    **Output Products:**
    - Binary cloud mask
    - Cloud coverage percentage
    - Clear area visualization
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Loading image data..."

        file_name = job_id + "_raw.npy"
        FILE_PATH = os.path.join(RAW_IMAGES_PATH, file_name)

        print(f"Saving raw image to {FILE_PATH}")
        print(f"Image shape: {image.shape}")

        if image.shape[2] != 13:
            image = image[:,:,:13] 

        np.save(FILE_PATH, image)

        metadata = preprocessing.extract_tiles_with_padding(FILE_PATH, job_id, (tiles_size, tiles_size, image.shape[2]), TILES_IMAGES_PATH)

        # Save metadata for debugging purposes
        metadata_path = os.path.join(METADATA_FOLDER, f"{job_id}_metadata.xlsx")
        metadata.to_excel(metadata_path, index=False)

        print("Images loaded and tiled")

        processing_jobs[job_id].message = "Running cloud detection..."

        for i in range(metadata.shape[0]):
            tile_path = os.path.join(TILES_IMAGES_PATH, f"{job_id}_tile_{metadata['tile_coordinates'][i]}.npy")
            
            result, cloud_mask, perc_cloudy = cloud_detection.is_cloudy(tile_path, job_id=job_id, cloud_threshold=0.5)

            # Save cloud_masks

            cloud_mask_path = os.path.join(CLOUD_IMAGES_PATH, f"{job_id}_{metadata['tile_coordinates'][i]}_cloud_mask.npy")
            np.save(cloud_mask_path, cloud_mask)

            cloudy = result['cloudy_tiles'] > 0

            metadata.at[i, 'cloud?'] = cloudy
            metadata.at[i, 'cloud_percentage'] = perc_cloudy

            processing_jobs[job_id].tiles_processed += 1
            processing_jobs[job_id].successful_tiles += result['clean_tiles']
            processing_jobs[job_id].progress = int((processing_jobs[job_id].tiles_processed / metadata.shape[0]) * 100)

        # Generate visualization images after all tiles are processed
        processing_jobs[job_id].message = "Generating visualization images..."

        metadata.to_excel(metadata_path, index=False)
        
        # Create RGB visualization
        plotting.create_rgb_visualization(image, DISPLAY_FOLDER, job_id=job_id)

        # Create cloud mask visualization
        plotting.create_cloud_mask_visualization(DATA_PATH, metadata_path, job_id=job_id)
        
        # Convert metadata to list of dictionaries for JSON serialization
        metadata_list = []
        for _, row in metadata.iterrows():
            metadata_list.append({
                "tile_id": f"tile_{row['tile_coordinates']}",
                "coordinates": row['tile_coordinates'],
                "cloud_coverage": row.get('cloud_percentage', 0),
                "is_cloudy": row.get('cloud?', False),
                "tile_path": f"{job_id}_tile_{row['tile_coordinates']}.npy"
            })
        
        # Store processed image metadata
        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="cloud_detection",
            rgb_image_url=f"{job_id}_rgb.png",
            cloud_image_url=f"{job_id}_cloud.png", 
            forest_image_url="", 
            heatmap_image_url="", 
            metadata=metadata_list,
            processing_time=processing_time
        )
        
        # Mark job as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].message = "Cloud detection completed successfully"
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"
        print(f"Error details: {str(e)}")

def process_forest_detection(job_id: str, cloud_job_id):
    """
    Background task for forest detection processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Calculate vegetation indices (NDVI, EVI, SAVI)
    3. Apply forest classification algorithms
    4. Identify forest boundaries and types
    5. Generate forest cover map
    
    **Algorithms Used:**
    - Vegetation index thresholding
    - Random Forest classification
    - Convolutional Neural Networks
    - Object-based image analysis
    
    **Output Products:**
    - Forest area boundaries
    - Forest type classification
    - Vegetation health assessment
    - Deforestation change detection
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        forest_detection_job_id[cloud_job_id] = job_id

        processing_jobs[job_id].status = "processing"

        processing_jobs[job_id].message = "Loading image data and metadata..."

        metadata_path = os.path.join(METADATA_FOLDER, f"{cloud_job_id}_metadata.xlsx")

        metadata = pd.read_excel(metadata_path)

        processing_jobs[job_id].message = "Processing image..."

        for i in range(metadata.shape[0]):

            cloudy = metadata['cloud?'][i]

            if not cloudy:

                tile_path = os.path.join(TILES_IMAGES_PATH, f"{cloud_job_id}_tile_{metadata['tile_coordinates'][i]}.npy")

                print(f"Processing tile: {tile_path}")

                tile = np.load(tile_path)

                print("Image loaded! Executing ndvi vegetation detector")

                success, is_forest, veg_percentage = forest_detection.ndvi_veg_detector(tile, job_id=job_id, file_name=f"{cloud_job_id}_tile_{metadata['tile_coordinates'][i]}.npy")

                print("Information about NDVI")
                print(f"  Valid: {success}")
                print(f"  Is Forest: {is_forest}")
                print(f"  Vegetation Percentage: {veg_percentage:.2f}%")

                processing_jobs[job_id].tiles_processed += 1
                processing_jobs[job_id].progress = int((processing_jobs[job_id].tiles_processed / metadata.shape[0]) * 100)

                if success:
                    if is_forest:
                        print(f"  Forest detected in {tile_path} ({veg_percentage:.2f}%)")
                        processing_jobs[job_id].successful_tiles += 1
                    else:
                        print(f"  No forest detected in {tile_path} ({veg_percentage:.2f}%)")
                else:
                    raise ValueError(f"Forest detection failed for tile {tile_path}")

                metadata.at[i, 'forest?'] = is_forest
                metadata.at[i, 'vegetation_percentage'] = veg_percentage

            else:
                print("Tile cloudy, not processing it")

        metadata.to_excel(metadata_path, index=False)

        print("Plotting forest picture")

        plotting.create_forest_picture(output_folder=DATA_PATH, metadata_path=metadata_path, job_id=job_id, cloud_job_id=cloud_job_id)

        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()

        print("Processing ended!")

        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].completed_at = datetime.now()
        processing_jobs[job_id].message = "Forest detection completed successfully"
        
        metadata_list = []
        for _, row in metadata.iterrows():
            metadata_list.append({
                "tile_id": f"tile_{row['tile_coordinates']}",
                "coordinates": str(row['tile_coordinates']),  # Convertir a string para serialización
                "cloud_coverage": float(row.get('cloud_percentage', 0)),
                "is_cloudy": bool(row.get('cloud?', False)),
                "vegetation_percentage": float(row.get('vegetation_percentage', 0)),
                "is_forest": bool(row.get('forest?', False)),
                "tile_path": f"{cloud_job_id}_tile_{row['tile_coordinates']}.npy"
            })

        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="forest_detection",
            rgb_image_url=f"{cloud_job_id}_rgb.png",
            cloud_image_url=f"{cloud_job_id}_cloud.png",
            forest_image_url=f"{job_id}_forest.png", # Figure it out how to add it to fire prediction
            heatmap_image_url=f"",
            metadata=metadata_list,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"

def process_fire_prediction(job_id: str, cloud_job_id: str):
    """
    Background task for fire risk prediction processing.
    
    **Processing Steps:**
    1. Load and validate satellite image
    2. Extract environmental features (vegetation, moisture, temperature)
    3. Apply fire risk prediction models
    4. Generate risk probability maps
    5. Identify high-risk zones
    6. Send alerts for critical areas
    
    **Algorithms Used:**
    - U-Net deep learning architecture
    - Random Forest ensemble methods
    - Weather integration models
    - Topographic analysis
    
    **Output Products:**
    - Fire risk probability heatmap
    - Risk level classifications (Low/Medium/High/Critical)
    - Vulnerable area identification
    - Early warning alerts
    
    **Alert System:**
    - Automatic email notifications for high-risk areas
    - Risk threshold monitoring
    - Emergency contact integration
    
    Args:
        job_id (str): Job identifier
        file (UploadFile): Input satellite image
    """
    try:
        processing_jobs[job_id].status = "processing"

        processing_jobs[job_id].message = "Loading image data and metadata..."

        metadata_path = os.path.join(METADATA_FOLDER, f"{cloud_job_id}_metadata.xlsx")

        metadata = pd.read_excel(metadata_path)

        processing_jobs[job_id].message = "Processing image..."

        for i in range(metadata.shape[0]):

            cloudy = metadata['cloud?'][i]

            processing_jobs[job_id].tiles_processed += 1
            processing_jobs[job_id].progress = int((processing_jobs[job_id].tiles_processed / metadata.shape[0]) * 100)

            if cloudy:
                continue

            tile_path = os.path.join(TILES_IMAGES_PATH, f"{cloud_job_id}_tile_{metadata['tile_coordinates'][i]}.npy")

            print(f"Processing tile: {tile_path}")

            tile = np.load(tile_path)

            # Calculate BANDS
            print("Calculating NDVI and NDMI...")

            nir = tile[..., 7].astype(np.float32)
            red = tile[..., 3].astype(np.float32)

            ndvi = np.divide(nir - red, nir + red,
                            out=np.zeros_like(nir), where=(nir + red) != 0)

            tile = np.concatenate((tile, ndvi[..., np.newaxis]), axis=-1)

            # Add NDMI index

            swir = tile[..., 11].astype(np.float32)
            nir = tile[..., 7].astype(np.float32)

            ndmi = np.divide(nir - swir, nir + swir,
                                out=np.zeros_like(nir), where=(nir + swir) != 0)

            tile = np.concatenate((tile, ndmi[..., np.newaxis]), axis=-1)

            # Process Fire Detection

            fire_mask, fire_prob = fire_prediction.predict_fire(tile)

            np.save(os.path.join(FIRE_IMAGES_PATH, f"{job_id}_tile_{metadata['tile_coordinates'][i]}_fire_mask.npy"), fire_mask)
            np.save(os.path.join(FIRE_IMAGES_PATH, f"{job_id}_tile_{metadata['tile_coordinates'][i]}_fire_prob.npy"), fire_prob)



        plotting.create_heatmap(DATA_PATH, metadata_path=metadata_path, job_id=job_id)

        processing_time = (datetime.now() - processing_jobs[job_id].created_at).total_seconds()

        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].message = "Processing completed successfully."

        metadata_list = []
        for _, row in metadata.iterrows():
            metadata_list.append({
                "tile_id": f"tile_{row['tile_coordinates']}",
                "coordinates": str(row['tile_coordinates']),  # Convertir a string para serialización
                "cloud_coverage": float(row.get('cloud_percentage', 0)),
                "is_cloudy": bool(row.get('cloud?', False)),
                "vegetation_percentage": float(row.get('vegetation_percentage', 0)),
                "is_forest": bool(row.get('forest?', False)),
                "tile_path": f"{cloud_job_id}_tile_{row['tile_coordinates']}.npy"
            })
        
        processed_images[job_id] = ProcessedImageResponse(
            job_id=job_id,
            task="fire_prediction",
            rgb_image_url=f"{cloud_job_id}_rgb.png",
            cloud_image_url=f"{cloud_job_id}_cloud.png",
            forest_image_url=f"{forest_detection_job_id[cloud_job_id]}_forest.png",
            heatmap_image_url=f"{job_id}_heatmap.png",
            metadata=metadata_list,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"


if __name__ == "__main__":
    # uvicorn.run("app:app", host="0.0.0.0", port=5001, workers=4)
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)