# auth.py
import os
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

def create_gmail_service():
    client_secret_file = os.getenv("GOOGLE_CLIENT_SECRET", "/app/secrets/client_secret.json")
    token_file = os.getenv("GOOGLE_TOKEN_FILE", "/app/secrets/token.json")

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Do this ONCE on a machine with a browser to generate token.json,
            # then mount/copy the token into the container. Browsers rarely work inside containers.
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=8080)
        with open(token_file, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)