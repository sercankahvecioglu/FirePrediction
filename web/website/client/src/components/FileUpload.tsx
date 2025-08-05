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
      'image/tiff': ['.tif'],
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
              <p className="text-lg font-medium text-white">Analiz ediliyor...</p>
              <p className="text-sm text-gray-400">Lütfen bekleyin</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Upload className="w-12 h-12 text-primary-400 mx-auto" />
            <div>
              <p className="text-lg font-medium text-white">
                {isDragActive ? 'Drop the file here' : 'Drag & Drop .tif or .npy files here'}
              </p>
              <p className="text-sm text-gray-400">or click to browse</p>
            </div>
            <div className="text-xs text-gray-500">
              Supported formats: .tif, .npy
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload; 