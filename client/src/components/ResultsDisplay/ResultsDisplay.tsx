import styles from './ResultsDisplay.module.css';

export type ResultsProps = {
  cloudUrl: string;
  forestUrl: string;
  fireUrl: string;
};

const ResultsDisplay = ({ cloudUrl, forestUrl, fireUrl }: ResultsProps) => {
  return (
    <section className={styles.section} aria-label="Processing results">
      <div className={styles.grid}>
        <article className={styles.card} aria-label="Cloud Mask">
          <h3 className={styles.cardTitle}>Cloud Mask</h3>
          <img src={cloudUrl} alt="Cloud detection mask" loading="lazy" />
        </article>
        <article className={styles.card} aria-label="Forest Mask">
          <h3 className={styles.cardTitle}>Forest Mask</h3>
          <img src={forestUrl} alt="Forest cover mask" loading="lazy" />
        </article>
        <article className={styles.card} aria-label="Fire Prediction">
          <h3 className={styles.cardTitle}>Fire Prediction</h3>
          <img src={fireUrl} alt="Fire prediction map" loading="lazy" />
        </article>
      </div>
    </section>
  );
};

export default ResultsDisplay;
