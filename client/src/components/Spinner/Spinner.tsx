import styles from './Spinner.module.css';

type SpinnerProps = {
  label?: string;
};

const Spinner = ({ label }: SpinnerProps) => {
  return (
    <div className={styles.wrapper} role="status" aria-live="polite">
      <div className={styles.spinner} aria-hidden="true" />
      {label && <span className={styles.label}>{label}</span>}
    </div>
  );
};

export default Spinner;
