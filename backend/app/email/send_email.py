import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def write_email(to_email, cc_emails, subject, path):
    # Read email configurations from environment variables
    sender_email = os.getenv('SENDER_EMAIL','vikram@signitysolutions.com')
    sender_password = os.getenv('SENDER_PASSWORD')
    smtp_server = os.getenv('SMTP_SERVER')
    smtp_port = int(os.getenv('SMTP_PORT', 587)) 

    # Construct email body
    body = f"Your inference results are ready at the following path:\n{path}"

    # Create message container
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['CC'] = ', '.join(cc_emails)
    msg['Subject'] = subject
   
    try:
        # Attach body to the message
        msg.attach(MIMEText(body, 'plain'))

        # Connect to SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)

        # Send email
        server.sendmail(sender_email, [to_email] + cc_emails, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"An error occurred while sending email: {e}")
        raise  
    finally:
        # Close the connection
        if 'server' in locals():
            server.quit()  # Close the connection if it was established



