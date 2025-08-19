import { useState, useCallback } from 'react';
import { predict } from '@/services/api';
import type { ProcessingState, ImageProcessingHook, PredictResponse } from '@/types/api';

const initialState: ProcessingState = {
  isProcessing: false,
  progress: 0,
  progressMessage: '',
  error: null,
  results: null,
};

export const useImageProcessing = (): ImageProcessingHook => {
  const [state, setState] = useState<ProcessingState>(initialState);

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  const processImage = useCallback(async (file: File) => {
    // Reset state and start processing
    setState({
      isProcessing: true,
      progress: 0,
      progressMessage: 'Initializing processing...',
      error: null,
      results: null,
    });

    try {
      const results = await predict(file, (progress, message) => {
        setState(prev => ({
          ...prev,
          progress,
          progressMessage: message,
        }));
      });

      // Processing completed successfully
      setState(prev => ({
        ...prev,
        isProcessing: false,
        progress: 100,
        progressMessage: 'Processing completed successfully!',
        results,
      }));
    } catch (error) {
      // Processing failed
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      setState(prev => ({
        ...prev,
        isProcessing: false,
        progress: 0,
        progressMessage: '',
        error: errorMessage,
      }));
      throw error; // Re-throw so the component can handle it (e.g., show toast)
    }
  }, []);

  return {
    ...state,
    processImage,
    reset,
  };
};
