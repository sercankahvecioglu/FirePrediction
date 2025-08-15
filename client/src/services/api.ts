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

  const jobResponse: JobResponse = await response.json();

  return jobResponse;
  
}

// Submit image for forest detection processing
async function submitImageForForestDetection(file: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/submit-image/forest-detection`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit image for forest detection');
  }

  return response.json();
}

// Submit image for fire prediction processing
async function submitImageForFirePrediction(file: File): Promise<JobResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/submit-image/fire-prediction`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit image for fire prediction');
  }

  return response.json();
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

    while(true) {

      const statusResponse = await fetch(`${API_BASE_URL}/job-status/${jobResponse.job_id}`);
      if (!statusResponse.ok) {
        const errorData = await statusResponse.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to get job status');
      }
      
      const status: JobStatus = await statusResponse.json();

      if(status.status === 'failed') {
        throw new Error(status.message || 'Processing failed');
      }
      else if (status.status === 'completed') {
        break; // Exit loop if processing is complete}
      }

      await pollJobStatus(jobResponse.job_id, (status) => {
      // Map backend progress (0-100) to our progress range (20-90)
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
      });

      onProgress?.(90, 'Retrieving processed images...');

      // Wait before polling again
      await new Promise(resolve => setTimeout(resolve, 2000));    
    }
    
    // Step 3: Get the final result
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

// Mock function for testing without real files
export async function predictMock(
  task: 'cloud_detection' | 'forest_detection' | 'fire_prediction' = 'cloud_detection',
  onProgress?: (progress: number, message: string) => void
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

// Process specific analysis types
export async function predictCloudDetection(
  file: File,
  onProgress?: (progress: number, message: string) => void
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
    
    onProgress?.(100, 'Cloud detection completed!');

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
    console.error('Cloud detection error:', error);
    throw error;
  }
}

export async function predictForestDetection(
  file: File,
  onProgress?: (progress: number, message: string) => void
): Promise<PredictResponse> {
  try {
    onProgress?.(10, 'Submitting image for forest detection...');
    const jobResponse = await submitImageForForestDetection(file);
    
    onProgress?.(20, 'Forest detection started...');
    await pollJobStatus(jobResponse.job_id, (status) => {
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
    });

    onProgress?.(90, 'Retrieving forest detection results...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    onProgress?.(100, 'Forest detection completed!');

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
    console.error('Forest detection error:', error);
    throw error;
  }
}

export async function predictFireRisk(
  file: File,
  onProgress?: (progress: number, message: string) => void
): Promise<PredictResponse> {
  try {
    onProgress?.(10, 'Submitting image for fire risk prediction...');
    const jobResponse = await submitImageForFirePrediction(file);
    
    onProgress?.(20, 'Fire risk prediction started...');
    await pollJobStatus(jobResponse.job_id, (status) => {
      const mappedProgress = 20 + Math.floor((status.progress / 100) * 70);
      onProgress?.(mappedProgress, status.message);
    });

    onProgress?.(90, 'Retrieving fire risk prediction results...');
    const result = await getProcessingResult(jobResponse.job_id);
    
    onProgress?.(100, 'Fire risk prediction completed!');

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
    console.error('Fire risk prediction error:', error);
    throw error;
  }
}
