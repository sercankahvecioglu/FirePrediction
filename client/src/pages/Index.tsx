import { useEffect, useState } from 'react';
import Header from '@/components/Header/Header';
import InfoSection from '@/components/InfoSection/InfoSection';
import ImageUploader from '@/components/ImageUploader/ImageUploader';
import ResultsDisplay from '@/components/ResultsDisplay/ResultsDisplay';
import { predict, type PredictResponse } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

const Index = () => {
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictResponse | null>(null);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  const { toast } = useToast();

  useEffect(() => {
    document.title = 'Flame Sentinels – Wildfire Prediction System';
    const meta = document.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute('content', 'Upload .npy files to generate RGB, cloud, forest, and fire prediction visualizations.');
  }, []);

  const handleUpload = async (file: File) => {
    setError(null);
    setProcessing(true);
    setResults(null);
    setProgress(0);
    setProgressMessage('');
    
    try {
      const data = await predict(file, (progressValue, message) => {
        setProgress(progressValue);
        setProgressMessage(message);
      });
      setResults(data);
      toast({ 
        title: 'Processing completed', 
        description: 'Your satellite image has been successfully analyzed.' 
      });
    } catch (e: any) {
      const message = e?.message || 'Error processing the file. Please try again.';
      setError(message);
      toast({ title: 'Processing error', description: message });
    } finally {
      setProcessing(false);
      setProgress(0);
      setProgressMessage('');
    }
  };

  return (
    <div>
      <Header />
      <main className="container mx-auto px-4 py-6">
        <section aria-labelledby="uploader-heading" className="mb-6">
          <h2 id="uploader-heading" className="sr-only">Upload</h2>
          <ImageUploader
            onUpload={handleUpload}
            onError={(msg) => { setError(msg); toast({ title: 'Invalid file', description: msg }); }}
            processing={processing}
            progress={progress}
            progressMessage={progressMessage}
            rgbUrl={results?.image_urls.rgb}
          />
          {error && (
            <p role="alert" className="mt-3 text-sm text-muted-foreground">{error}</p>
          )}
        </section>

        {results && (
          <ResultsDisplay
            cloudUrl={results.image_urls.cloud}
            forestUrl={results.image_urls.forest}
            fireUrl={results.image_urls.fire}
          />)
        }
        <InfoSection />
      </main>
    </div>
  );
};

export default Index;
