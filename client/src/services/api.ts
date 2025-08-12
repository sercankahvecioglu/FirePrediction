// Types based on the backend API response models
export interface JobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface JobStatus {
  job_id: string;
  task: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  tiles_to_process: number;
  tiles_processed: number;
  successful_tiles: number;
  progress: number;
  message: string;
}

export interface ProcessedImageResponse {
  job_id: string;
  task: string;
  rgb_image_url: string;
  cloud_image_url: string;
  forest_image_url: string;
  heatmap_image_url: string;
  metadata: any;
}

export interface PredictResponse {
  job_id: string;
  status: 'success' | 'error';
  image_urls: {
    rgb: string;
    cloud: string;
    forest: string;
    fire: string;
  };
}

const API_BASE_URL = '/api';

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

// Main function that handles the complete flow
export async function predict(
  file: File, 
  onProgress?: (progress: number, message: string) => void
): Promise<PredictResponse> {
  try {
    // Step 1: Submit image for cloud detection
    onProgress?.(10, 'Submitting image for processing...');
    const jobResponse = await submitImageForCloudDetection(file);
    
    // Step 2: Poll for completion with progress updates
    onProgress?.(20, 'Processing started, waiting for completion...');
    
    await pollJobStatus(jobResponse.job_id, (status) => {
      // Map backend progress (0-100) to our progress range (20-90)
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
    });

    // Step 3: Get the final result
    onProgress?.(90, 'Retrieving processed images...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    onProgress?.(100, 'Processing completed successfully!');

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
    console.error('Prediction error:', error);
    throw error;
  }
}
