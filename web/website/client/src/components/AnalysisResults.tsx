import React, { useState } from 'react';
import { Eye } from 'lucide-react';

interface AnalysisData {
  coordinates: {
    latitude: number;
    longitude: number;
  };
  timestamp: string;
  resolution: string;
  riskAssessment: {
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    totalArea: number;
  };
  heatmapData: any[];
  confidence: number;
}

interface AnalysisResultsProps {
  data: AnalysisData;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ data }) => {
  const [riskThreshold, setRiskThreshold] = useState(50);

  // Mock satellite image (gerçek uygulamada bu bir API'den gelecek)
  const satelliteImageUrl = 'https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=800&h=600&fit=crop';

  // Mock heatmap (gerçek uygulamada bu analiz sonucundan gelecek)
  const heatmapUrl = 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800&h=600&fit=crop';

  return (
    <div className="space-y-8">
      <h2 className="text-3xl font-bold text-center text-white mb-8">
        Analiz Sonuçları
      </h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Original Satellite Image */}
        <div className="card">
          <h3 className="text-xl font-semibold text-white mb-4">Original Satellite Image</h3>
          <div className="relative">
            <img 
              src={satelliteImageUrl} 
              alt="Satellite Image" 
              className="w-full h-64 object-cover rounded-lg"
            />
          </div>
          <div className="mt-4 space-y-2 text-sm text-gray-300">
            <p>Koordinatlar: {data.coordinates.latitude}° N, {data.coordinates.longitude}° W</p>
            <p>Zaman: {new Date(data.timestamp).toLocaleString('tr-TR')}</p>
            <p>Çözünürlük: {data.resolution}</p>
          </div>
        </div>

        {/* Fire Risk Heatmap */}
        <div className="card">
          <h3 className="text-xl font-semibold text-white mb-4">Fire Risk Heatmap</h3>
          <div className="relative">
            <img 
              src={heatmapUrl} 
              alt="Risk Heatmap" 
              className="w-full h-64 object-cover rounded-lg"
            />
          </div>
          
          {/* Risk Threshold Slider */}
          <div className="mt-4">
            <div className="flex items-center space-x-2 mb-2">
              <Eye className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Risk Eşiği: {riskThreshold}%</span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0"
                max="100"
                value={riskThreshold}
                onChange={(e) => setRiskThreshold(Number(e.target.value))}
                className="w-full h-2 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0%</span>
                <span>25%</span>
                <span>50%</span>
                <span>75%</span>
                <span>100%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults; 