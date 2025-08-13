import { useCallback, useRef, useState } from 'react';
import Spinner from '../Spinner/Spinner';
import styles from './ImageUploader.module.css';

export type ImageURLs = {
  rgb: string;
  cloud: string;
  forest: string;
  fire: string;
};

type ImageUploaderProps = {
  onUpload: (file: File) => void;
  onError: (message: string) => void;
  processing: boolean;
  progress?: number;
  progressMessage?: string;
  rgbUrl?: string | null;
};

const ALLOWED_EXT = '.npy';

const ImageUploader = ({ onUpload, onError, processing, progress = 0, progressMessage = '', rgbUrl }: ImageUploaderProps) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const validateFile = (file: File) => {
    return file && file.name.toLowerCase().endsWith(ALLOWED_EXT);
  };

  const handleFiles = useCallback(
    (file?: File) => {
      if (!file) return;
      if (!validateFile(file)) {
        onError('Invalid file type. Please upload a .npy file.');
        return;
        }
      onUpload(file);
    },
    [onError, onUpload]
  );

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    handleFiles(file);
  };

  const onDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!dragging) setDragging(true);
  };

  const onDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const onSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    handleFiles(file);
  };

  const onMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    e.currentTarget.style.setProperty('--x', `${x}%`);
    e.currentTarget.style.setProperty('--y', `${y}%`);
  };

  return (
    <section className={styles.section} aria-label="Upload wildfire data file">
      <div
        className={`${styles.dropZone} ${dragging ? styles.dragging : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onMouseMove={onMouseMove}
      >
        {!processing && !rgbUrl && (
          <>
            <p className={styles.prompt}>Drag and drop a .npy file here</p>
            <p className={styles.or}>or</p>
            <button
              type="button"
              className={styles.cta}
              onClick={() => inputRef.current?.click()}
            >
              Choose .npy file
            </button>
            <input
              ref={inputRef}
              type="file"
              accept=".npy"
              onChange={onSelect}
              className={styles.input}
              aria-label="Choose .npy file"
            />
            <p className={styles.hint}>Only .npy files are accepted.</p>
          </>
        )}

        {processing && (
          <div className={styles.processing}>
            <Spinner label="Processing image..." />
            {progressMessage && (
              <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                <p style={{ fontSize: '0.9rem', color: '#666', marginBottom: '0.5rem' }}>
                  {progressMessage}
                </p>
                <div style={{ 
                  width: '100%', 
                  backgroundColor: '#e0e0e0', 
                  borderRadius: '4px', 
                  overflow: 'hidden',
                  height: '8px'
                }}>
                  <div 
                    style={{
                      width: `${progress}%`,
                      height: '100%',
                      backgroundColor: '#4CAF50',
                      transition: 'width 0.3s ease'
                    }}
                  />
                </div>
                <p style={{ fontSize: '0.8rem', color: '#888', marginTop: '0.5rem' }}>
                  {progress}% complete
                </p>
              </div>
            )}
          </div>
        )}

        {!processing && rgbUrl && (
          <figure className={styles.preview}>
            <img
              src={rgbUrl}
              alt="Wildfire RGB visualization"
              loading="eager"
            />
            <figcaption className={styles.caption}>Main RGB image</figcaption>
          </figure>
        )}
      </div>
    </section>
  );
};

export default ImageUploader;
