import styles from './InfoSection.module.css';

const InfoSection = () => {
  return (
    <section className={styles.section} aria-labelledby="about-heading">
      <div className={styles.card}>
        <h2 id="about-heading" className={styles.title}>About Flame Sentinels</h2>
        <p className={styles.lead}>
          Flame Sentinels helps analysts and responders visualize wildfire insights from satellite-based data.
          Upload a .npy or .tiff file to generate an RGB view, cloud and forest masks, and a final fire prediction.
        </p>

        <div className={styles.grid}> 
          <article className={styles.block}>
            <h3 className={styles.blockTitle}>How it works</h3>
            <ul className={styles.list}>
              <li>Upload your .npy or .tiff file and start processing.</li>
              <li>We run the model and return images showing the result of the processing.</li>
              <li>View the RGB image and detailed result cards.</li>
            </ul>
          </article>

          <article className={styles.block}>
            <h3 className={styles.blockTitle}>Supported file type</h3>
            <p className={styles.text}>We accept .npy and .tiff files for processing.</p>
          </article>

          <article className={styles.block}>
            <h3 className={styles.blockTitle}>Privacy</h3>
            <p className={styles.text}>Your file is processed solely to generate predictions and images for your session.</p>
          </article>
        </div>
      </div>
    </section>
  );
};

export default InfoSection;
