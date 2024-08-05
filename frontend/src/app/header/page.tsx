import Image from 'next/image';
import styles from 'src/app/page.module.css';

const Header = () => {
  return (
    <header className={styles.header}>
      <Image src="/logo-white.png" alt="Envisify" width={150} height={150} />
       

    </header>
  );
};

export default Header;
