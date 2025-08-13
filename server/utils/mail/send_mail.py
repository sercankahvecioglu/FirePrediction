# send_mail.py
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from .auth import create_gmail_service
from datetime import datetime

def send_mail(subject="Flame Sentinels Alert", recipient="sercankahvecioglu123@gmail.com", 
              high_risk_percentage=None, average_risk=None, job_id=None):
    """
    Send fire risk alert email.
    
    Args:
        subject (str): Email subject line
        recipient (str): Recipient email address
        high_risk_percentage (float): Percentage of high-risk tiles
        average_risk (float): Average risk score
        job_id (str): Job identifier for reference
    """
    service = create_gmail_service()

    # Build email message with risk details
    if high_risk_percentage is not None and average_risk is not None:
        emailMsg = f"""Flame Sentinels High Fire Risk Alert

URGENT: High risk of fire detected in the analyzed satellite image.

Risk Analysis Summary:
- High-risk tiles: {high_risk_percentage:.1f}%
- Average risk score: {average_risk:.3f}/1.0
- Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Job ID: {job_id or 'N/A'}

Immediate Actions Recommended:
1. Review the detailed risk heatmap
2. Alert local authorities if applicable
3. Implement fire prevention measures in high-risk areas
4. Monitor weather conditions closely

Please take necessary precautions and contact emergency services if immediate action is required.

This is an automated alert from the Flame Sentinels Fire Prediction System."""
    else:
        emailMsg = "Flame Sentinels Alert: High risk of fire in the area. Please take necessary precautions."
    
    mimeMessage = MIMEMultipart()
    mimeMessage["to"] = recipient
    mimeMessage["subject"] = subject
    mimeMessage.attach(MIMEText(emailMsg, "plain"))

    raw = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

    # Send email
    try:
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        print("Fire risk alert email sent. Message Id:", sent["id"])
        return sent["id"]
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise
