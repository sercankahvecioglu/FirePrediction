import styles from './Header.module.css';

const Header = () => {
  return (
    <header className={styles.header} role="banner">
      <div className={styles.container}>
        <div className={styles.brand}>
          <div className={styles.logoSlot} aria-label="Espacio para el logo" />
          <h1 className={styles.title}>Flame Sentinels</h1>
        </div>
      </div>
    </header>
  );
};

export default Header;
