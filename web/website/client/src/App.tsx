import React, { useState } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import FeatureHighlights from './components/FeatureHighlights';
import AnalysisResults from './components/AnalysisResults';
import RiskAssessment from './components/RiskAssessment';
import Footer from './components/Footer';

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

function App() {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    setIsAnalyzing(true);
    
    try {
      // Dosya yükleme
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadResponse = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        throw new Error('Dosya yükleme hatası');
      }
      
      const uploadResult = await uploadResponse.json();
      
      // Analiz başlat (fileId ile)
      const analyzeResponse = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ fileId: uploadResult.file.id }),
      });
      
      if (!analyzeResponse.ok) {
        throw new Error('Analiz hatası');
      }
      
      const analysisResult = await analyzeResponse.json();
      setAnalysisData(analysisResult.results);
    } catch (error) {
      console.error('Hata:', error);
      alert('İşlem sırasında bir hata oluştu');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark-900">
      <Header />
      
      <main className="container mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold mb-4">
            <span className="text-primary-400">Wildfire Risk Intelligence</span>
            <span className="text-green-400"> Platform</span>
          </h1>
          <p className="text-xl text-gray-300 mb-8">
            AI-Powered Fire Risk Prediction from Satellite Imagery
          </p>
        </div>

        {/* File Upload Section */}
        <div className="mb-12">
          <FileUpload onFileUpload={handleFileUpload} isAnalyzing={isAnalyzing} />
        </div>

        {/* Feature Highlights */}
        <div className="mb-12">
          <FeatureHighlights />
        </div>

        {/* Analysis Results */}
        {analysisData && (
          <div className="mb-12">
            <AnalysisResults data={analysisData} />
          </div>
        )}

        {/* Risk Assessment */}
        {analysisData && (
          <div className="mb-12">
            <RiskAssessment data={analysisData.riskAssessment} />
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App; 