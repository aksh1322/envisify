import React from 'react';
import styles from 'src/app/page.module.css'

function page() {
  return (
    <div>
     <footer className={styles.footer}>
      <div className={styles.footerSection}>
        <h2>Terms & Conditions</h2>
        <ul>
          <li><a >Privacy Policy</a></li>
          <li><a >Cookie Policy</a></li>
        </ul>
      </div>
      <div className={styles.footerSection}>
        <h2>About</h2>
        <ul>
          <li><a>Our Team</a></li>
          <li><a>Company History</a></li>
        </ul>
      </div>
      <div className={styles.footerSection}>
        <h2>Contact Us</h2>
        <ul>
          <li><a>Email</a></li>
          <li><a>Phone</a></li>
        </ul>
      </div>
    </footer>
    </div>
  )
}

export default page