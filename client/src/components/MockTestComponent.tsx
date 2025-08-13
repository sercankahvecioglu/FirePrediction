import React, { useState } from 'react';
import { predictMock, predictCloudDetection, predictForestDetection, predictFireRisk } from '../services/api';

interface MockTestComponentProps {
  className?: string;
}

interface TestResults {
  job_id?: string;
  image_urls?: {
    rgb: string;
    cloud: string;
    forest: string;
    fire: string;
  };
  error?: string;
}

/**
 * MockTestComponent - A testing component for mock data functionality
 * 
 * This component demonstrates how to use the mock data endpoints for testing
 * the satellite image analysis workflow without requiring real files.
 * 
 * Usage:
 * 1. Click any "Test" button to start mock processing
 * 2. Watch the progress updates in real-time
 * 3. View the generated mock images when complete
 * 
 * TODO: Remove this component in production - it's for testing only
 */
export const MockTestComponent: React.FC<MockTestComponentProps> = ({ className }) => {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [results, setResults] = useState<TestResults>({});
  const [activeTest, setActiveTest] = useState<string>('');

  const resetState = () => {
    setProgress(0);
    setStatusMessage('');
    setResults({});
  };

  const handleProgressUpdate = (progress: number, message: string) => {
    setProgress(progress);
    setStatusMessage(message);
  };

  const runMockTest = async (testType: 'cloud_detection' | 'forest_detection' | 'fire_prediction') => {
    setLoading(true);
    setActiveTest(testType);
    resetState();

    try {
      const result = await predictMock(testType, handleProgressUpdate);
      setResults(result);
      setStatusMessage('Mock test completed successfully!');
    } catch (error) {
      console.error('Mock test failed:', error);
      setResults({ error: error instanceof Error ? error.message : 'Unknown error occurred' });
      setStatusMessage('Mock test failed');
    } finally {
      setLoading(false);
      setActiveTest('');
    }
  };

  const runRealTest = async (testType: 'cloud' | 'forest' | 'fire') => {
    // For testing with file upload - requires actual file
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.npy,.tiff,.tif';
    
    fileInput.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      setLoading(true);
      setActiveTest(testType);
      resetState();

      try {
        let result;
        switch (testType) {
          case 'cloud':
            result = await predictCloudDetection(file, handleProgressUpdate);
            break;
          case 'forest':
            result = await predictForestDetection(file, handleProgressUpdate);
            break;
          case 'fire':
            result = await predictFireRisk(file, handleProgressUpdate);
            break;
        }
        setResults(result);
        setStatusMessage('Real processing completed successfully!');
      } catch (error) {
        console.error('Real test failed:', error);
        setResults({ error: error instanceof Error ? error.message : 'Unknown error occurred' });
        setStatusMessage('Real processing failed');
      } finally {
        setLoading(false);
        setActiveTest('');
      }
    };

    fileInput.click();
  };

  return (
    <div className={`p-6 bg-white rounded-lg shadow-lg ${className}`}>
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Mock Data Testing</h2>
      
      {/* Mock Test Buttons */}
      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <h3 className="font-semibold text-blue-800 mb-2">Cloud Detection</h3>
          <p className="text-sm text-gray-600 mb-3">
            Tests cloud detection with synthetic satellite imagery and cloud masks
          </p>
          <button
            onClick={() => runMockTest('cloud_detection')}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors"
          >
            Test Cloud Detection
          </button>
        </div>

        <div className="bg-green-50 p-4 rounded-lg">
          <h3 className="font-semibold text-green-800 mb-2">Forest Detection</h3>
          <p className="text-sm text-gray-600 mb-3">
            Tests forest classification with mock vegetation indices and forest types
          </p>
          <button
            onClick={() => runMockTest('forest_detection')}
            disabled={loading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors"
          >
            Test Forest Detection
          </button>
        </div>

        <div className="bg-red-50 p-4 rounded-lg">
          <h3 className="font-semibold text-red-800 mb-2">Fire Prediction</h3>
          <p className="text-sm text-gray-600 mb-3">
            Tests fire risk prediction with synthetic environmental data and risk heatmaps
          </p>
          <button
            onClick={() => runMockTest('fire_prediction')}
            disabled={loading}
            className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors"
          >
            Test Fire Prediction
          </button>
        </div>
      </div>

      {/* Real Processing Test Buttons */}
      <div className="border-t pt-4 mb-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Real File Processing (Upload Required)</h3>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => runRealTest('cloud')}
            disabled={loading}
            className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors text-sm"
          >
            Test with Real File - Cloud
          </button>
          <button
            onClick={() => runRealTest('forest')}
            disabled={loading}
            className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors text-sm"
          >
            Test with Real File - Forest
          </button>
          <button
            onClick={() => runRealTest('fire')}
            disabled={loading}
            className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white px-4 py-2 rounded transition-colors text-sm"
          >
            Test with Real File - Fire
          </button>
        </div>
      </div>

      {/* Progress Display */}
      {loading && (
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-gray-700">Processing Progress</span>
            <span className="text-sm text-gray-500">{progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          {statusMessage && (
            <p className="text-sm text-gray-600 mt-2">{statusMessage}</p>
          )}
          {activeTest && (
            <p className="text-sm text-blue-600 mt-1">
              Active test: {activeTest.replace('_', ' ')}
            </p>
          )}
        </div>
      )}

      {/* Results Display */}
      {results.error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <h4 className="font-semibold text-red-800 mb-2">Error</h4>
          <p className="text-red-700">{results.error}</p>
        </div>
      )}

      {results.image_urls && (
        <div className="bg-gray-50 border rounded-lg p-4">
          <h4 className="font-semibold text-gray-800 mb-4">Processing Results</h4>
          
          {results.job_id && (
            <p className="text-sm text-gray-600 mb-4">Job ID: {results.job_id}</p>
          )}

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {results.image_urls.rgb && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">RGB Image</h5>
                <img
                  src={results.image_urls.rgb}
                  alt="RGB visualization"
                  className="w-full h-32 object-cover rounded border"
                  onError={(e) => {
                    e.currentTarget.src = '/placeholder.svg';
                  }}
                />
              </div>
            )}

            {results.image_urls.cloud && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Cloud Detection</h5>
                <img
                  src={results.image_urls.cloud}
                  alt="Cloud detection mask"
                  className="w-full h-32 object-cover rounded border"
                  onError={(e) => {
                    e.currentTarget.src = '/placeholder.svg';
                  }}
                />
              </div>
            )}

            {results.image_urls.forest && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Forest Classification</h5>
                <img
                  src={results.image_urls.forest}
                  alt="Forest classification map"
                  className="w-full h-32 object-cover rounded border"
                  onError={(e) => {
                    e.currentTarget.src = '/placeholder.svg';
                  }}
                />
              </div>
            )}

            {results.image_urls.fire && (
              <div>
                <h5 className="font-medium text-gray-700 mb-2">Fire Risk Heatmap</h5>
                <img
                  src={results.image_urls.fire}
                  alt="Fire risk prediction heatmap"
                  className="w-full h-32 object-cover rounded border"
                  onError={(e) => {
                    e.currentTarget.src = '/placeholder.svg';
                  }}
                />
              </div>
            )}
          </div>

          <div className="mt-4 p-3 bg-blue-50 rounded">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> These are mock-generated images for testing purposes. 
              In production, replace with real satellite image analysis algorithms.
            </p>
          </div>
        </div>
      )}

      {/* Integration Instructions */}
      <div className="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
        <h4 className="font-semibold text-yellow-800 mb-2">Integration Notes</h4>
        <ul className="text-sm text-yellow-700 space-y-1">
          <li>• Mock endpoints generate synthetic data for testing workflow</li>
          <li>• Replace with real algorithms from utils/ modules for production</li>
          <li>• Cloud detection: Integrate utils.cloud_detection.clouddetector</li>
          <li>• Forest detection: Use utils.forest_detection modules with NDVI/EVI</li>
          <li>• Fire prediction: Implement utils.fire_prediction.unet models</li>
          <li>• Email alerts: Configure utils.mail.send_mail for fire warnings</li>
        </ul>
      </div>
    </div>
  );
};

export default MockTestComponent;
