# send_mail.py
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from auth import create_gmail_service

def send_mail():
    service = create_gmail_service()

    # Build MIME message
    emailMsg = "Flame Sentinels Alert : High risk of fire in the area. Please take necessary precautions."
    mimeMessage = MIMEMultipart()
    mimeMessage["to"] = "sercankahvecioglu123@gmail.com"
    mimeMessage["subject"] = "Flame Sentinels Alert"
    mimeMessage.attach(MIMEText(emailMsg, "plain"))

    raw = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

    # --- Option A: send ---
    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    print("Message sent. Id:", sent["id"])
