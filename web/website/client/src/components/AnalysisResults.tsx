import React, { useState } from 'react';
import { Eye } from 'lucide-react';

interface AnalysisData {
  success: boolean;
  message: string;
  timestamp: string;
}

interface AnalysisResultsProps {
  data: AnalysisData;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ data }) => {
  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-center text-white mb-8">
        Backend Yanıtı
      </h2>
      
      <div className="card">
        <div className="text-center space-y-4">
          <div className={`text-2xl font-bold ${data.success ? 'text-green-400' : 'text-red-400'}`}>
            {data.success ? '✅ Başarılı' : '❌ Hata'}
          </div>
          <p className="text-lg text-white">{data.message}</p>
          <p className="text-sm text-gray-400">
            Zaman: {new Date(data.timestamp).toLocaleString('tr-TR')}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults; 