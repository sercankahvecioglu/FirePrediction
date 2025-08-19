/**
 * API Response Types for Satellite Image Processing Pipeline
 * 
 * This file defines the TypeScript interfaces for the FastAPI backend
 * that handles satellite image processing with the sequential pipeline:
 * Cloud Detection → Forest Detection → Fire Prediction
 */

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
  progress: number; // 0-100
  message: string;
}

export interface ProcessedImageResponse {
  job_id: string;
  task: string;
  rgb_image_url: string;
  cloud_image_url: string;
  forest_image_url: string;
  heatmap_image_url: string;
  metadata: any[];
  processing_time?: number;
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

/**
 * Processing State for the useImageProcessing hook
 */
export interface ProcessingState {
  isProcessing: boolean;
  progress: number;
  progressMessage: string;
  error: string | null;
  results: PredictResponse | null;
}

/**
 * Hook interface for image processing
 */
export interface ImageProcessingHook extends ProcessingState {
  processImage: (file: File) => Promise<void>;
  reset: () => void;
}

/**
 * Image URLs structure for components
 */
export interface ImageURLs {
  rgb: string;
  cloud: string;
  forest: string;
  fire: string;
}

/**
 * Processing pipeline step
 */
export type PipelineStep = 'cloud_detection' | 'forest_detection' | 'fire_prediction';

/**
 * Progress callback function type
 */
export type ProgressCallback = (progress: number, message: string) => void;
