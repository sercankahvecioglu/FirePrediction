// Types based on the backend API response models
import type { 
  JobResponse, 
  JobStatus, 
  ProcessedImageResponse, 
  PredictResponse,
  ProgressCallback
} from '@/types/api';

const API_BASE_URL = 'http://localhost:5001';

// Submit image for cloud detection processing
async function submitImageForCloudDetection(file: File, tileSize: number = 256): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('tile_size', tileSize.toString());

  const response = await fetch(`${API_BASE_URL}/submit-image/cloud-detection`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit image for processing');
  }

  return response.json();
}

// Submit for forest detection using cloud job ID (sequential pipeline)
async function submitForestDetection(cloudJobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/submit-image/forest-detection?cloud_job_id=${cloudJobId}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit forest detection');
  }

  return response.json();
}

// Submit for fire prediction using cloud job ID (sequential pipeline)
async function submitFirePrediction(cloudJobId: string): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/submit-image/fire-prediction?cloud_job_id=${cloudJobId}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit fire prediction');
  }

  return response.json();
}

// Download image by name
async function downloadImage(imageName: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/download-picture/${imageName}`);
  
  if (!response.ok) {
    throw new Error(`Failed to download image: ${imageName}`);
  }

  const blob = await response.blob();
  return URL.createObjectURL(blob);
}

// Mock endpoints for testing without real files
async function submitMockTask(task: 'cloud_detection' | 'forest_detection' | 'fire_prediction'): Promise<JobResponse> {
  const response = await fetch(`${API_BASE_URL}/mock-submit-image/${task}`, {
    method: 'POST',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to submit mock ${task} task`);
  }

  return response.json();
}

// Check job status
async function getJobStatus(jobId: string): Promise<JobStatus> {
  const response = await fetch(`${API_BASE_URL}/job-status/${jobId}`);
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to get job status');
  }

  return response.json();
}

// Get processing result
async function getProcessingResult(jobId: string): Promise<ProcessedImageResponse> {
  const response = await fetch(`${API_BASE_URL}/get-result/${jobId}`);
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to get processing result');
  }

  return response.json();
}

// Poll job status until completion
async function pollJobStatus(jobId: string, onProgress?: (status: JobStatus) => void): Promise<JobStatus> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        
        if (onProgress) {
          onProgress(status);
        }

        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.message || 'Processing failed'));
        } else {
          // Continue polling every 2 seconds
          setTimeout(poll, 2000);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    poll();
  });
}

// Main function that handles the complete sequential pipeline
export async function predict(
  file: File, 
  onProgress?: ProgressCallback
): Promise<PredictResponse> {
  try {
    // Step 1: Submit image for cloud detection
    onProgress?.(5, 'Submitting image for cloud detection...');
    const cloudJobResponse = await submitImageForCloudDetection(file);
    
    // Step 2: Wait for cloud detection to complete
    onProgress?.(10, 'Processing cloud detection...');
    await pollJobStatus(cloudJobResponse.job_id, (status) => {
      // Map progress to 10-35% range for cloud detection
      const mappedProgress = 10 + Math.floor((status.progress / 100) * 25);
      onProgress?.(mappedProgress, `Cloud detection: ${status.message}`);
    });

    // Step 3: Submit for forest detection using cloud job ID
    onProgress?.(35, 'Starting forest detection...');
    const forestJobResponse = await submitForestDetection(cloudJobResponse.job_id);
    
    // Step 4: Wait for forest detection to complete
    onProgress?.(40, 'Processing forest detection...');
    await pollJobStatus(forestJobResponse.job_id, (status) => {
      // Map progress to 40-65% range for forest detection
      const mappedProgress = 40 + Math.floor((status.progress / 100) * 25);
      onProgress?.(mappedProgress, `Forest detection: ${status.message}`);
    });

    // Step 5: Submit for fire prediction using cloud job ID
    onProgress?.(65, 'Starting fire prediction...');
    const fireJobResponse = await submitFirePrediction(cloudJobResponse.job_id);
    
    // Step 6: Wait for fire prediction to complete
    onProgress?.(70, 'Processing fire prediction...');
    await pollJobStatus(fireJobResponse.job_id, (status) => {
      // Map progress to 70-90% range for fire prediction
      const mappedProgress = 70 + Math.floor((status.progress / 100) * 20);
      onProgress?.(mappedProgress, `Fire prediction: ${status.message}`);
    });

    // Step 7: Get the final result from fire prediction job
    onProgress?.(90, 'Retrieving processed images...');
    const result = await getProcessingResult(fireJobResponse.job_id);
    
    // Step 8: Download all images and create blob URLs
    onProgress?.(95, 'Downloading images...');
    const [rgbUrl, cloudUrl, forestUrl, fireUrl] = await Promise.all([
      downloadImage(result.rgb_image_url),
      downloadImage(result.cloud_image_url),
      downloadImage(result.forest_image_url),
      downloadImage(result.heatmap_image_url)
    ]);
    
    onProgress?.(100, 'Processing completed successfully!');

    return {
      job_id: result.job_id,
      status: 'success',
      image_urls: {
        rgb: rgbUrl,
        cloud: cloudUrl,
        forest: forestUrl,
        fire: fireUrl,
      },
    };
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

// Mock function for testing without real files
export async function predictMock(
  task: 'cloud_detection' | 'forest_detection' | 'fire_prediction' = 'cloud_detection',
  onProgress?: ProgressCallback
): Promise<PredictResponse> {
  try {
    // Step 1: Submit mock task
    onProgress?.(10, `Submitting mock ${task.replace('_', ' ')} task...`);
    const jobResponse = await submitMockTask(task);
    
    // Step 2: Poll for completion with progress updates
    onProgress?.(20, 'Mock processing started, waiting for completion...');
    
    await pollJobStatus(jobResponse.job_id, (status) => {
      // Map backend progress (0-100) to our progress range (20-90)
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
    });

    // Step 3: Get the final result
    onProgress?.(90, 'Retrieving mock processed images...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    onProgress?.(100, 'Mock processing completed successfully!');

    // Convert backend response to frontend format
    return {
      job_id: result.job_id,
      status: 'success',
      image_urls: {
        rgb: result.rgb_image_url,
        cloud: result.cloud_image_url,
        forest: result.forest_image_url,
        fire: result.heatmap_image_url,
      },
    };
  } catch (error) {
    console.error('Mock prediction error:', error);
    throw error;
  }
}

// Process specific analysis types - Updated for new pipeline
export async function predictCloudDetection(
  file: File,
  onProgress?: ProgressCallback
): Promise<PredictResponse> {
  try {
    onProgress?.(10, 'Submitting image for cloud detection...');
    const jobResponse = await submitImageForCloudDetection(file);
    
    onProgress?.(20, 'Cloud detection started...');
    await pollJobStatus(jobResponse.job_id, (status) => {
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
    });

    onProgress?.(90, 'Retrieving cloud detection results...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    // Download images and create blob URLs
    const [rgbUrl, cloudUrl, forestUrl, fireUrl] = await Promise.all([
      downloadImage(result.rgb_image_url),
      downloadImage(result.cloud_image_url),
      downloadImage(result.forest_image_url),
      downloadImage(result.heatmap_image_url)
    ]);
    
    onProgress?.(100, 'Cloud detection completed!');

    return {
      job_id: result.job_id,
      status: 'success',
      image_urls: {
        rgb: rgbUrl,
        cloud: cloudUrl,
        forest: forestUrl,
        fire: fireUrl,
      },
    };
  } catch (error) {
    console.error('Cloud detection error:', error);
    throw error;
  }
}

export async function predictForestDetection(
  file: File,
  onProgress?: ProgressCallback
): Promise<PredictResponse> {
  try {
    // First run cloud detection
    onProgress?.(10, 'Running cloud detection first...');
    const cloudJobResponse = await submitImageForCloudDetection(file);
    
    await pollJobStatus(cloudJobResponse.job_id, (status) => {
      const mappedProgress = 10 + Math.floor((status.progress / 100) * 30);
      onProgress?.(mappedProgress, `Cloud detection: ${status.message}`);
    });

    onProgress?.(40, 'Submitting forest detection...');
    const jobResponse = await submitForestDetection(cloudJobResponse.job_id);
    
    onProgress?.(50, 'Forest detection started...');
    await pollJobStatus(jobResponse.job_id, (status) => {
      const mappedProgress = 50 + Math.floor((status.progress / 100) * 40);
      onProgress?.(mappedProgress, status.message);
    });

    onProgress?.(90, 'Retrieving forest detection results...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    // Download images and create blob URLs
    const [rgbUrl, cloudUrl, forestUrl, fireUrl] = await Promise.all([
      downloadImage(result.rgb_image_url),
      downloadImage(result.cloud_image_url),
      downloadImage(result.forest_image_url),
      downloadImage(result.heatmap_image_url)
    ]);
    
    onProgress?.(100, 'Forest detection completed!');

    return {
      job_id: result.job_id,
      status: 'success',
      image_urls: {
        rgb: rgbUrl,
        cloud: cloudUrl,
        forest: forestUrl,
        fire: fireUrl,
      },
    };
  } catch (error) {
    console.error('Forest detection error:', error);
    throw error;
  }
}

export async function predictFireRisk(
  file: File,
  onProgress?: ProgressCallback
): Promise<PredictResponse> {
  try {
    // This is the same as the main predict function since fire risk needs the full pipeline
    return await predict(file, onProgress);
  } catch (error) {
    console.error('Fire risk prediction error:', error);
    throw error;
  }
}
