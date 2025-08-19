import { useEffect } from 'react';
import Header from '@/components/Header/Header';
import InfoSection from '@/components/InfoSection/InfoSection';
import ImageUploader from '@/components/ImageUploader/ImageUploader';
import ResultsDisplay from '@/components/ResultsDisplay/ResultsDisplay';
import { useImageProcessing } from '@/hooks/useImageProcessing';
import { useToast } from '@/hooks/use-toast';

const Index = () => {
  const { 
    isProcessing, 
    progress, 
    progressMessage, 
    error, 
    results, 
    processImage, 
    reset 
  } = useImageProcessing();
  const { toast } = useToast();

  useEffect(() => {
    document.title = 'Flame Sentinels – Wildfire Prediction System';
    const meta = document.querySelector('meta[name="description"]');
    if (meta) meta.setAttribute('content', 'Upload .npy files to generate RGB, cloud, forest, and fire prediction visualizations.');
  }, []);

  const handleUpload = async (file: File) => {
    try {
      await processImage(file);
      toast({ 
        title: 'Processing completed', 
        description: 'Your satellite image has been successfully analyzed.' 
      });
    } catch (e: any) {
      const message = e?.message || 'Error processing the file. Please try again.';
      toast({ 
        title: 'Processing error', 
        description: message,
        variant: 'destructive'
      });
    }
  };

  const handleError = (message: string) => {
    toast({ 
      title: 'Invalid file', 
      description: message,
      variant: 'destructive'
    });
  };

  return (
    <div>
      <Header />
      <main className="container mx-auto px-4 py-6">
        <section aria-labelledby="uploader-heading" className="mb-6">
          <h2 id="uploader-heading" className="sr-only">Upload</h2>
          <ImageUploader
            onUpload={handleUpload}
            onError={handleError}
            processing={isProcessing}
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
