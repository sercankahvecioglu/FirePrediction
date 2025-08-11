import React, { useState } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import FeatureHighlights from './components/FeatureHighlights';
import AnalysisResults from './components/AnalysisResults';
import RiskAssessment from './components/RiskAssessment';
import Footer from './components/Footer';


interface AnalysisData {
  success: boolean;
  message: string;
  timestamp: string;
}

function App() {
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const apiUrl = process.env.REACT_APP_API_URL;

  const testConnection = async () => {
  const formData = new FormData();

  // Send a dummy file (simulate a test image)
  const blob = new Blob(["test"], { type: "image/png" });
  formData.append("file", blob, "test.png");

  try {
    const response = await fetch(`/api/upload-image`, {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    console.log("Backend response:", result);
    alert(`Connected! Backend says: ${JSON.stringify(result)}`);
  } catch (err) {
    console.error("Failed to connect to backend:", err);
    alert("Failed to connect to backend.");
  }
};

  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    
    try {
      // Dosya yükleme
      const formData = new FormData();
      formData.append('file', file);
      
      const uploadResponse = await fetch(`/api/upload-image`, {
        method: 'POST',
        body: formData,
      });
      
      if (!uploadResponse.ok) {
        throw new Error('Dosya yükleme hatası');
      }
      const uploadResult = await uploadResponse.json();

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
        <div className="text-center mb-8">
        <button
          onClick={testConnection}
          className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
        >
          Test Backend Connection
        </button>
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


      </main>

      <Footer />
    </div>
  );
}

export default App; 