import styles from './Header.module.css';
import logo from '../../logo.jpeg';

const Header = () => {
  return (
    <header className={styles.header} role="banner">
      <div className={styles.container}>
        <div className={styles.brand}>
          <div className={styles.logoSlot} aria-label="Logo de Flame Sentinels">
            <img 
              src={logo} 
              alt="Flame Sentinels Logo" 
              className={styles.logo}
            />
          </div>
          <h1 className={styles.title}>Flame Sentinels</h1>
        </div>
      </div>
    </header>
  );
};

export default Header;
