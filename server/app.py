from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import uuid
import os
import json
from datetime import datetime
import asyncio
import sys

# ---- HOMEMADE LIBRARIES ---- #

from utils.mail.send_mail import send_mail # Sending mail adverting higher fire risk

# Load from utils library
from utils import cloud_detection
from utils import fire_prediction  
from utils import preprocessing
from utils import forest_detection
from utils import data_api 

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Satellite Image Fire Risk Prediction API",
    description="API for processing satellite images to predict fire risk using cloud masking and vegetation detection",
    version="1.0.0"
)

# Pydantic models for request/response schemas
class TileInfo(BaseModel):
    """Information about a single tile"""
    tile_id: str
    position: Tuple[int, int]  # (row, col) position in grid
    has_clouds: bool
    cloud_percentage: float
    is_discarded: bool
    vegetation_detected: bool
    fire_risk_score: float

class ProcessingStatus(BaseModel):
    """Status of image processing job"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime]
    total_tiles: int
    processed_tiles: int
    discarded_tiles: int

class FireRiskHeatmap(BaseModel):
    """Fire risk heatmap response"""
    job_id: str
    grid_dimensions: Tuple[int, int]  # (rows, cols)
    tiles: List[TileInfo]
    heatmap_url: str
    metadata: Dict

# In-memory storage for demo purposes (use database in production)
processing_jobs = {}
tile_data = {}

@app.get("/", tags=["Health Check"])
def read_root():
    """
    Health check endpoint to verify API is running.
    
    Returns:
        dict: Welcome message and API status
    """
    return {
        "message": "Welcome to the Satellite Image Fire Risk Prediction API",
        "status": "operational",
        "version": "1.0.0"
    }

@app.post("/upload-and-tile", response_model=ProcessingStatus, tags=["Image Processing"])
async def upload_and_tile_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tile_size: int = 256 
):
    """
    Upload a large satellite image and divide it into tiles for processing.
    
    This endpoint:
    1. Receives a large satellite image file
    2. Validates the image format and size
    3. Divides the image into smaller tiles of specified size
    4. Creates a processing job and returns job ID for tracking
    5. Starts background processing of tiles
    
    Args:
        file (UploadFile): The satellite image file (NPY supported at the moment)
        tile_size (int): Size of each tile in pixels (default: 256x256)

    Returns:
        ProcessingStatus: Job information including job_id for tracking progress
        
    Raises:
        HTTPException: If file format is unsupported or file is too large
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Validate file format
    # allowed_formats = [".tiff", ".tif", ".npy"]
    allowed_formats = [".npy"]
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Allowed: {allowed_formats}")
    
    # TODO: Implement image tiling logic here
    # - Save uploaded file to temporary location
    # - Load image using PIL/OpenCV/rasterio
    # - Validate image dimensions and format
    # - Calculate number of tiles needed
    # - Create tile grid and save individual tiles
    # - Store tile metadata (position, file paths, etc.)
    
    # Mock calculation for demo
    mock_total_tiles = 64  # This would be calculated based on image size and tile_size
    
    # Create processing job
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="pending",
        created_at=datetime.now(),
        completed_at=None,
        total_tiles=mock_total_tiles,
        processed_tiles=0,
        discarded_tiles=0
    )
    
    # Start background processing
    background_tasks.add_task(process_tiles_pipeline, job_id)
    
    return processing_jobs[job_id]

@app.get("/job-status/{job_id}", response_model=ProcessingStatus, tags=["Job Management"])
def get_job_status(job_id: str):
    """
    Get the current status of a processing job.
    
    This endpoint allows clients to check the progress of their image processing job,
    including how many tiles have been processed and current status.
    
    Args:
        job_id (str): Unique identifier for the processing job
        
    Returns:
        ProcessingStatus: Current job status and progress information
        
    Raises:
        HTTPException: If job_id is not found
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]

@app.post("/cloud-masking/{job_id}", tags=["Cloud Processing"])
async def run_cloud_masking(job_id: str):
    """
    Run cloud masking on all tiles for a specific job.
    
    This endpoint:
    1. Processes each tile to detect cloud coverage
    2. Calculates cloud percentage for each tile
    3. Marks tiles as discarded if cloud coverage exceeds threshold
    4. Updates tile status and cloud information
    5. Saves cleaned (cloud-free) tiles for further processing
    
    Args:
        job_id (str): Job identifier to process tiles for
        
    Returns:
        dict: Summary of cloud masking results including number of clean/discarded tiles
        
    Raises:
        HTTPException: If job not found or job is not in correct status
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "processing":
        raise HTTPException(status_code=400, detail="Job must be in processing status")
    
    # TODO: Implement cloud masking logic here
    # - Load each tile image
    # - Apply cloud detection algorithm (e.g., threshold-based, ML model)
    # - Calculate cloud percentage per tile
    # - Apply cloud mask to remove cloudy pixels
    # - Mark tiles as discarded if cloud coverage > threshold (e.g., 30%)
    # - Save cleaned tiles and update tile metadata
    # - Integration point: Use utils/clouddetector.py functions
    
    # Mock processing results
    clean_tiles = []
    discarded_tiles = []
    
    # Simulate cloud masking results
    for i in range(job.total_tiles):
        tile_id = f"{job_id}_tile_{i}"
        # Mock cloud detection results
        cloud_percentage = 25.0 if i % 4 != 0 else 75.0  # Mock: every 4th tile is very cloudy
        is_discarded = cloud_percentage > 50.0
        
        tile_info = TileInfo(
            tile_id=tile_id,
            position=(i // 8, i % 8),  # Assuming 8x8 grid
            has_clouds=cloud_percentage > 5.0,
            cloud_percentage=cloud_percentage,
            is_discarded=is_discarded,
            vegetation_detected=False,  # Will be set in vegetation detection step
            fire_risk_score=0.0  # Will be set in fire risk prediction step
        )
        
        if is_discarded:
            discarded_tiles.append(tile_info)
        else:
            clean_tiles.append(tile_info)
        
        tile_data[tile_id] = tile_info
    
    # Update job status
    job.discarded_tiles = len(discarded_tiles)
    job.processed_tiles = len(clean_tiles)
    
    return {
        "job_id": job_id,
        "total_tiles": job.total_tiles,
        "clean_tiles": len(clean_tiles),
        "discarded_tiles": len(discarded_tiles),
        "cloud_masking_complete": True,
        "clean_tile_ids": [tile.tile_id for tile in clean_tiles],
        "discarded_tile_ids": [tile.tile_id for tile in discarded_tiles]
    }

@app.post("/vegetation-detection/{job_id}", tags=["Vegetation Analysis"])
async def detect_vegetation(job_id: str):
    """
    Detect vegetation presence on clean (cloud-free) tiles.
    
    This endpoint:
    1. Processes only clean tiles (non-discarded) from cloud masking step
    2. Applies vegetation detection algorithms (e.g., NDVI, ML models)
    3. Updates tile metadata with vegetation presence information
    4. Prepares tiles for fire risk prediction
    
    Args:
        job_id (str): Job identifier to process vegetation detection for
        
    Returns:
        dict: Vegetation detection results including tiles with/without vegetation
        
    Raises:
        HTTPException: If job not found or cloud masking not completed
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get clean tiles for this job
    job_tiles = [tile for tile_id, tile in tile_data.items() if tile_id.startswith(job_id) and not tile.is_discarded]
    
    if not job_tiles:
        raise HTTPException(status_code=400, detail="No clean tiles available. Run cloud masking first.")
    
    # TODO: Implement vegetation detection logic here
    # - Load clean tile images
    # - Calculate vegetation indices (NDVI, EVI, etc.)
    # - Apply vegetation classification algorithms
    # - Determine vegetation presence/absence and type
    # - Update tile metadata with vegetation information
    # - Integration point: Use existing vegetation detection functions
    
    vegetation_tiles = []
    non_vegetation_tiles = []
    
    # Mock vegetation detection
    for tile in job_tiles:
        # Mock: assume 70% of clean tiles have vegetation
        has_vegetation = hash(tile.tile_id) % 10 < 7
        tile.vegetation_detected = has_vegetation
        
        if has_vegetation:
            vegetation_tiles.append(tile)
        else:
            non_vegetation_tiles.append(tile)
    
    return {
        "job_id": job_id,
        "processed_tiles": len(job_tiles),
        "vegetation_tiles": len(vegetation_tiles),
        "non_vegetation_tiles": len(non_vegetation_tiles),
        "vegetation_detection_complete": True,
        "vegetation_tile_ids": [tile.tile_id for tile in vegetation_tiles],
        "non_vegetation_tile_ids": [tile.tile_id for tile in non_vegetation_tiles]
    }

@app.post("/fire-risk-prediction/{job_id}", response_model=FireRiskHeatmap, tags=["Fire Risk Analysis"])
async def predict_fire_risk(job_id: str):
    """
    Predict fire risk on processed tiles and generate heatmap for the entire image.
    
    This endpoint:
    1. Processes tiles that have completed cloud masking and vegetation detection
    2. Applies fire risk prediction models to each clean tile
    3. Generates fire risk scores (0.0 to 1.0 scale)
    4. Creates a heatmap visualization for the entire original image
    5. Uses black squares for discarded tiles in the heatmap
    6. Returns heatmap URL and detailed metadata
    
    Args:
        job_id (str): Job identifier to generate fire risk prediction for
        
    Returns:
        FireRiskHeatmap: Complete fire risk analysis including heatmap URL and metadata
        
    Raises:
        HTTPException: If job not found or previous steps not completed
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Get all tiles for this job
    job_tiles = [tile for tile_id, tile in tile_data.items() if tile_id.startswith(job_id)]
    
    if not job_tiles:
        raise HTTPException(status_code=400, detail="No tiles found. Complete previous processing steps first.")
    
    # TODO: Implement fire risk prediction logic here
    # - Load processed tile data (clean tiles with vegetation info)
    # - Apply fire risk prediction models (ML models, rule-based systems)
    # - Consider factors: vegetation type, moisture, weather data, terrain
    # - Generate risk scores for each tile
    # - Integration point: Use FirePredictionModel/train_test.py and unet.py
    
    # TODO: Implement heatmap generation logic here
    # - Create grid representing original image tile layout
    # - Assign colors based on fire risk scores (e.g., green=low, red=high)
    # - Use black color for discarded tiles
    # - Generate heatmap image and save to static files
    # - Return URL to access heatmap image
    
    # Mock fire risk prediction
    high_risk_tiles_count = 0
    for tile in job_tiles:
        if not tile.is_discarded and tile.vegetation_detected:
            # Higher risk for vegetation tiles
            tile.fire_risk_score = min(0.8, max(0.2, hash(tile.tile_id) % 100 / 100.0))
        elif not tile.is_discarded:
            # Lower risk for non-vegetation tiles
            tile.fire_risk_score = min(0.3, max(0.0, hash(tile.tile_id) % 50 / 100.0))
        else:
            # No score for discarded tiles
            tile.fire_risk_score = 0.0
        
        # Count high risk tiles
        if tile.fire_risk_score > 0.7:
            high_risk_tiles_count += 1
    
    # Mock heatmap generation
    heatmap_filename = f"heatmap_{job_id}.png"
    heatmap_url = f"/download-heatmap/{job_id}"
    
    # Calculate grid dimensions (assuming square grid for simplicity)
    total_tiles = len(job_tiles)
    grid_size = int(total_tiles ** 0.5)
    grid_dimensions = (grid_size, grid_size)
    
    # Update job as completed
    processing_jobs[job_id].status = "completed"
    processing_jobs[job_id].completed_at = datetime.now()
    
    # Generate metadata
    metadata = {
        "processing_date": datetime.now().isoformat(),
        "total_tiles": len(job_tiles),
        "clean_tiles": len([t for t in job_tiles if not t.is_discarded]),
        "discarded_tiles": len([t for t in job_tiles if t.is_discarded]),
        "vegetation_tiles": len([t for t in job_tiles if t.vegetation_detected]),
        "high_risk_tiles": len([t for t in job_tiles if t.fire_risk_score > 0.7]),
        "medium_risk_tiles": len([t for t in job_tiles if 0.3 < t.fire_risk_score <= 0.7]),
        "low_risk_tiles": len([t for t in job_tiles if 0.0 < t.fire_risk_score <= 0.3]),
        "average_risk_score": sum(t.fire_risk_score for t in job_tiles if not t.is_discarded) / len([t for t in job_tiles if not t.is_discarded]) if job_tiles else 0
    }
    
    # Send email alert if high fire risk is detected
    # High risk threshold: more than 20% of tiles have high risk score (>0.7) or average risk > 0.6
    high_risk_percentage = (metadata["high_risk_tiles"] / metadata["clean_tiles"]) * 100 if metadata["clean_tiles"] > 0 else 0
    average_risk = metadata["average_risk_score"]
    
    if high_risk_percentage > 20 or average_risk > 0.6:
        try:
            # TODO: Customize email content based on specific risk analysis
            # - Include location information if available
            # - Add risk level details and recommendations
            # - Attach heatmap image if generated
            send_mail(
                subject="🔥 URGENT: High Fire Risk Detected - Flame Sentinels Alert",
                high_risk_percentage=high_risk_percentage,
                average_risk=average_risk,
                job_id=job_id
            )
            print(f"High fire risk alert sent for job {job_id}: {high_risk_percentage:.1f}% high-risk tiles, avg risk: {average_risk:.3f}")
        except Exception as e:
            print(f"Failed to send fire risk alert email: {e}")
            # Don't fail the entire request if email fails
    
    return FireRiskHeatmap(
        job_id=job_id,
        grid_dimensions=grid_dimensions,
        tiles=job_tiles,
        heatmap_url=heatmap_url,
        metadata=metadata
    )

@app.get("/download-heatmap/{job_id}", tags=["File Download"])
async def download_heatmap(job_id: str):
    """
    Download the generated fire risk heatmap image.
    
    This endpoint serves the generated heatmap image file for a completed job.
    The heatmap shows fire risk levels across the original image with:
    - Color coding for risk levels (green=low, yellow=medium, red=high)
    - Black squares for discarded tiles (high cloud coverage)
    
    Args:
        job_id (str): Job identifier to download heatmap for
        
    Returns:
        FileResponse: The heatmap image file
        
    Raises:
        HTTPException: If job not found or heatmap not generated
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Heatmap not ready. Complete fire risk prediction first.")
    
    # TODO: Implement actual file serving logic here
    # - Check if heatmap file exists
    # - Return FileResponse with proper headers
    # - Handle file not found cases
    
    heatmap_path = f"static/heatmaps/heatmap_{job_id}.png"
    
    # For demo purposes, return a placeholder response
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail="Heatmap file not found")
    
    return FileResponse(
        path=heatmap_path,
        media_type="image/png",
        filename=f"fire_risk_heatmap_{job_id}.png"
    )

@app.get("/tile-details/{tile_id}", response_model=TileInfo, tags=["Tile Management"])
def get_tile_details(tile_id: str):
    """
    Get detailed information about a specific tile.
    
    This endpoint provides comprehensive information about a single tile including:
    - Position in the original image grid
    - Cloud coverage information
    - Vegetation detection results
    - Fire risk score
    - Processing status
    
    Args:
        tile_id (str): Unique identifier for the tile
        
    Returns:
        TileInfo: Detailed tile information
        
    Raises:
        HTTPException: If tile not found
    """
    if tile_id not in tile_data:
        raise HTTPException(status_code=404, detail="Tile not found")
    
    return tile_data[tile_id]

@app.delete("/job/{job_id}", tags=["Job Management"])
def delete_job(job_id: str):
    """
    Delete a processing job and all associated data.
    
    This endpoint:
    1. Removes job from processing queue
    2. Deletes all associated tile data
    3. Removes generated files (tiles, heatmaps, etc.)
    4. Cleans up temporary storage
    
    Args:
        job_id (str): Job identifier to delete
        
    Returns:
        dict: Deletion confirmation message
        
    Raises:
        HTTPException: If job not found
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # TODO: Implement cleanup logic here
    # - Delete tile files from storage
    # - Remove heatmap files
    # - Clean up temporary directories
    # - Remove from database (if using persistent storage)
    
    # Remove from in-memory storage
    del processing_jobs[job_id]
    
    # Remove associated tile data
    tile_ids_to_remove = [tile_id for tile_id in tile_data.keys() if tile_id.startswith(job_id)]
    for tile_id in tile_ids_to_remove:
        del tile_data[tile_id]
    
    return {"message": f"Job {job_id} and all associated data deleted successfully"}

@app.get("/jobs", response_model=List[ProcessingStatus], tags=["Job Management"])
def list_all_jobs():
    """
    List all processing jobs with their current status.
    
    This endpoint provides an overview of all jobs in the system,
    useful for monitoring and administration purposes.
    
    Returns:
        List[ProcessingStatus]: List of all jobs with their status information
    """
    return list(processing_jobs.values())

async def process_tiles_pipeline(job_id: str):
    """
    Background task to orchestrate the complete tile processing pipeline.
    
    This function:
    1. Updates job status to "processing"
    2. Coordinates the execution of cloud masking, vegetation detection, and fire risk prediction
    3. Handles errors and updates job status accordingly
    4. Can be extended to include email notifications or webhooks
    
    Args:
        job_id (str): Job identifier to process
    """
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        
        # TODO: Implement complete pipeline orchestration here
        # - Call cloud masking functions
        # - Call vegetation detection functions  
        # - Call fire risk prediction functions
        # - Handle intermediate results and error states
        # - Update progress incrementally
        # - Send notifications on completion/failure
        
        # Simulate processing time
        await asyncio.sleep(2)
        
        # For demo purposes, keep job in processing state
        # In real implementation, this would coordinate all processing steps
        
    except Exception as e:
        # Handle processing errors
        processing_jobs[job_id].status = "failed"
        # TODO: Log error details and notify user

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
