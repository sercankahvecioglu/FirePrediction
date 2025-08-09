import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, Loader2 } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  isAnalyzing: boolean;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, isAnalyzing }) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/tiff': ['.tif', '.tiff'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'application/octet-stream': ['.npy']
    },
    multiple: false
  });

  return (
    <div className="max-w-2xl mx-auto">
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
          ${isDragActive 
            ? 'border-primary-400 bg-primary-900/20' 
            : 'border-gray-600 hover:border-primary-400 hover:bg-dark-800'
          }
          ${isAnalyzing ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {isAnalyzing ? (
          <div className="space-y-4">
            <Loader2 className="w-12 h-12 text-primary-400 animate-spin mx-auto" />
            <div>
              <p className="text-lg font-medium text-white">Analyzing...</p>
              <p className="text-sm text-gray-400">Please wait while we process your image</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="w-12 h-12 text-primary-400 mx-auto" />
            <div>
              <p className="text-lg font-medium text-white">
                {isDragActive ? 'Drop the file here' : 'Drag & Drop your image here'}
              </p>
              <p className="text-sm text-gray-400">or click to browse</p>
            </div>
            <div className="text-xs text-gray-500">
              Supported formats: .png, .jpg, .jpeg, .tif, .tiff, .npy
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;