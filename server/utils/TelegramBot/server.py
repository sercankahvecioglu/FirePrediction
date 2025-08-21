import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
import threading
import time
import asyncio
import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7572088114:AAFvD5SVexAynrLUEzl1hjvXDY35UGnqz34")
# Use relative path from the server directory
VIDEO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", "krasi_aurafarming.mp4")

# Configure logging to see errors
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# List of users subscribed to alerts
subscribers = set()
application = None

# Memory system for last fire alert
last_fire_alert = {
    "image_path": None,
    "message": "No fire analysis available yet.",
    "timestamp": None
}

# /start command: tells you to use /last_fire
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_message = """🔥 **Welcome to Fire Prediction Alert System from Flame Sentinels Team!** 🔥

This bot provides real-time fire risk analysis and alerts based on satellite imagery and AI predictions.

**Available Commands:**
🔍 `/last_fire` - View the latest fire risk analysis with heatmap
🔔 `/set_alert` - Subscribe to receive fire risk notifications

**How it works:**
• Our AI analyzes satellite data to detect fire risks
• You'll receive alerts when significant risks are detected
• Images show risk level heatmaps for affected areas

Get started by using `/last_fire` to see the current analysis, or `/set_alert` to receive automatic notifications!"""
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_message,
        parse_mode='Markdown'
    )

# /last_fire command: shows the latest fire analysis with image
async def last_fire(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if last_fire_alert["image_path"] and last_fire_alert["message"]:
        try:
            # Check if image file exists
            if os.path.exists(last_fire_alert["image_path"]):
                # Send image with message using local file path
                with open(last_fire_alert["image_path"], 'rb') as image_file:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=image_file,
                        caption=last_fire_alert["message"]
                    )
            else:
                logging.error(f"Image file not found: {last_fire_alert['image_path']}")
                # Fallback to text only
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=last_fire_alert["message"] + "\n\n⚠️ Image file not available."
                )
        except Exception as e:
            logging.error(f"Error sending fire image: {e}")
            # Fallback to text only
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=last_fire_alert["message"]
            )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=last_fire_alert["message"]
        )

# /set_alert command: subscribes the user to alerts
async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    subscribers.add(user_id)
    
    confirmation_message = """🔔 **Alert Subscription Activated!** ✅

You have successfully subscribed to fire risk alerts. Here's what you can expect:

**What you'll receive:**
• 🔴 Critical alerts for immediate attention
• 📊 Heatmaps showing affected areas

**Alert frequency:**
• Only when significant risks are detected
• Real-time notifications based on satellite analysis
• No spam - only important safety information

You can use `/last_fire` anytime to check the current fire risk status.

Stay safe! 🛡️"""
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=confirmation_message,
        parse_mode='Markdown'
    )

async def secret(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if user_id in subscribers and application:
        try:
            # Read video from path
            with open(VIDEO_PATH, 'rb') as video:
                message = "You found the secret video! Happy aura farming!"
                await context.bot.send_video(
                    chat_id=user_id, 
                    video=video, 
                    caption=message
                )
        except Exception as e:
            logging.error(f"Error sending secret video to {user_id}: {e}")
            await context.bot.send_message(
                chat_id=user_id,
                text="Secret video not available right now, but you found it! 🎉"
            )
    else:
        await context.bot.send_message(
            chat_id=user_id,
            text="You need to subscribe to alerts first with /set_alert to access this feature!"
        )

# ============ EXTERNAL API FUNCTIONS ============

def update_last_fire_alert(image_path, message):
    """
    Update the last fire alert with image path and message
    
    Args:
        image_path (str): Local path to the fire analysis image
        message (str): The fire analysis message
    """
    global last_fire_alert

    print("Updating info!")
    
    last_fire_alert = {
        "image_path": image_path,
        "message": message,
        "timestamp": datetime.now()
    }
    logging.info(f"Last fire alert updated: {message}")
    logging.info(f"Image path: {image_path}")

def send_fire_alert(image_path, message):
    """
    Send a fire alert with image to all subscribed users
    
    Args:
        image_path (str): Local path to the fire analysis image
        message (str): The alert message to send
    """
    # Update the last fire alert memory
    update_last_fire_alert(image_path, message)
    
    if subscribers and application:
        def send_alerts():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def send_alert_to_user(user_id):
                    try:
                        # Check if image file exists
                        if os.path.exists(image_path):
                            # Send image with message using local file path
                            with open(image_path, 'rb') as image_file:
                                await application.bot.send_photo(
                                    chat_id=user_id,
                                    photo=image_file,
                                    caption=message
                                )
                        else:
                            logging.error(f"Image file not found: {image_path}")
                            # Fallback to text only
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=message + "\n\n⚠️ Image file not available."
                            )
                    except Exception as e:
                        logging.error(f"Error sending alert to {user_id}: {e}")
                        # Fallback to text only
                        try:
                            await application.bot.send_message(
                                chat_id=user_id,
                                text=message
                            )
                        except Exception as e2:
                            logging.error(f"Error sending fallback message to {user_id}: {e2}")
                
                async def send_to_all_users():
                    for user_id in subscribers:
                        await send_alert_to_user(user_id)
                
                # Run the async function
                loop.run_until_complete(send_to_all_users())
                loop.close()
                    
            except Exception as e:
                logging.error(f"Error in send_alerts: {e}")
        
        # Run in a separate thread to avoid blocking
        alert_thread = threading.Thread(target=send_alerts)
        alert_thread.start()
        logging.info(f"Fire alert sent to {len(subscribers)} subscribers")
    else:
        logging.warning("No subscribers or application not initialized")

def get_subscribers():
    """
    Get the list of subscribed user IDs
    
    Returns:
        set: Set of subscribed user IDs
    """
    return subscribers.copy()

def add_subscriber(user_id):
    """
    Add a user to the subscribers list
    
    Args:
        user_id (int): The user ID to add
    """
    subscribers.add(user_id)
    logging.info(f"User {user_id} added to subscribers")

def remove_subscriber(user_id):
    """
    Remove a user from the subscribers list
    
    Args:
        user_id (int): The user ID to remove
    """
    subscribers.discard(user_id)
    logging.info(f"User {user_id} removed from subscribers")

def is_bot_running():
    """
    Check if the bot application is running
    
    Returns:
        bool: True if bot is running, False otherwise
    """
    return application is not None

def get_last_fire_alert():
    """
    Get the last fire alert information
    
    Returns:
        dict: Last fire alert with image_path, message, and timestamp
    """
    return last_fire_alert.copy()

def get_subscriber_count():
    """
    Get the number of subscribed users
    
    Returns:
        int: Number of subscribers
    """
    return len(subscribers)

def send_fire_alert_with_risk_level(image_path, job_id, risk_level="MEDIUM", custom_message=None):
    """
    Send a fire alert with predefined message based on risk level or custom message
    
    Args:
        image_path (str): Local path to the fire analysis image
        job_id (str): Job ID for the analysis
        risk_level (str): Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        custom_message (str): Optional custom message. If provided, overrides risk level messages
    """
    if custom_message:
        # Use custom message if provided
        message = custom_message
        message += f"\n\nAnalysis ID: {job_id}"
        message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        # Use predefined messages based on risk level
        risk_messages = {
            "LOW": "🟢 Fire Risk: LOW - No immediate threats detected in the analyzed area.",
            "MEDIUM": "🟡 Fire Risk: MEDIUM - Some elevated conditions detected. Monitor conditions carefully.",
            "HIGH": "🟠 Fire Risk: HIGH - Significant fire risk detected! Enhanced monitoring recommended.",
            "CRITICAL": "🔴 Fire Risk: CRITICAL - Extreme fire danger! Immediate action may be required!"
        }
        
        message = risk_messages.get(risk_level.upper(), risk_messages["MEDIUM"])
        message += f"\n\nAnalysis ID: {job_id}"
        message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Send alert if risk level is MEDIUM or higher, or if custom message is provided
    if custom_message or risk_level.upper() in ["MEDIUM", "HIGH", "CRITICAL"]:
        send_fire_alert(image_path, message)
        logging.info(f"Fire alert sent with {'custom message' if custom_message else 'risk level: ' + risk_level}")
    else:
        # Just update the last fire alert for LOW risk
        update_last_fire_alert(image_path, message)
        logging.info(f"Last fire alert updated with risk level: {risk_level}")

def send_custom_fire_alert(image_path, custom_message, job_id=None):
    """
    Send a fire alert with completely custom message
    
    Args:
        image_path (str): Local path to the fire analysis image
        custom_message (str): Custom alert message
        job_id (str): Optional job ID to include in the message
    """
    message = custom_message
    
    if job_id:
        message += f"\n\nAnalysis ID: {job_id}"
    message += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    send_fire_alert(image_path, message)
    logging.info(f"Custom fire alert sent: {custom_message[:50]}...")

def send_simple_alert(message_text):
    """
    Send a simple text alert to all subscribers (no image)
    
    Args:
        message_text (str): The alert message to send
    """
    if subscribers and application:
        def send_text_alerts():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def send_text_to_user(user_id):
                    try:
                        await application.bot.send_message(
                            chat_id=user_id,
                            text=message_text
                        )
                    except Exception as e:
                        logging.error(f"Error sending text alert to {user_id}: {e}")
                
                async def send_to_all_users():
                    for user_id in subscribers:
                        await send_text_to_user(user_id)
                
                # Run the async function
                loop.run_until_complete(send_to_all_users())
                loop.close()
                    
            except Exception as e:
                logging.error(f"Error in send_text_alerts: {e}")
        
        alert_thread = threading.Thread(target=send_text_alerts)
        alert_thread.start()
        logging.info(f"Text alert sent to {len(subscribers)} subscribers")
    else:
        logging.warning("No subscribers or application not initialized")

# ============ END EXTERNAL API FUNCTIONS ============

# ============ HTTP API FOR EXTERNAL COMMUNICATION ============

# Create FastAPI app for HTTP communication
telegram_api = FastAPI(title="Telegram Bot API", version="1.0.0")

class AlertRequest(BaseModel):
    """Request model for fire alerts"""
    image_path: str
    message: str
    job_id: str = None

class SimpleAlertRequest(BaseModel):
    """Request model for simple text alerts"""
    message: str

@telegram_api.post("/send-fire-alert")
async def api_send_fire_alert(alert: AlertRequest):
    """
    API endpoint to receive fire alerts from external services
    
    Args:
        alert (AlertRequest): Alert data including image path and message
        
    Returns:
        dict: Status and subscriber count
    """
    try:
        send_fire_alert(alert.image_path, alert.message)
        return {
            "status": "success",
            "message": "Fire alert sent successfully",
            "subscribers_count": len(subscribers),
            "job_id": alert.job_id
        }
    except Exception as e:
        logging.error(f"Error in API send fire alert: {e}")
        return {
            "status": "error",
            "message": f"Failed to send alert: {str(e)}",
            "subscribers_count": len(subscribers)
        }

@telegram_api.post("/send-fire-alert-with-risk")
async def api_send_fire_alert_with_risk(
    image_path: str,
    job_id: str,
    risk_level: str = "MEDIUM",
    custom_message: str = None
):
    """
    API endpoint to send fire alerts with risk level
    
    Args:
        image_path (str): Path to the fire analysis image
        job_id (str): Job ID for the analysis
        risk_level (str): Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        custom_message (str): Optional custom message
        
    Returns:
        dict: Status and alert information
    """
    try:
        send_fire_alert(image_path, custom_message)
        send_fire_alert_with_risk_level(image_path, job_id, risk_level, custom_message)
        return {
            "status": "success",
            "message": f"Fire alert sent with risk level: {risk_level}",
            "subscribers_count": len(subscribers),
            "job_id": job_id,
            "risk_level": risk_level
        }
    except Exception as e:
        logging.error(f"Error in API send fire alert with risk: {e}")
        return {
            "status": "error",
            "message": f"Failed to send alert: {str(e)}",
            "subscribers_count": len(subscribers)
        }

@telegram_api.post("/send-simple-alert")
async def api_send_simple_alert(alert: SimpleAlertRequest):
    """
    API endpoint to send simple text alerts
    
    Args:
        alert (SimpleAlertRequest): Simple text alert data
        
    Returns:
        dict: Status and subscriber count
    """
    try:
        send_simple_alert(alert.message)
        return {
            "status": "success",
            "message": "Simple alert sent successfully",
            "subscribers_count": len(subscribers)
        }
    except Exception as e:
        logging.error(f"Error in API send simple alert: {e}")
        return {
            "status": "error",
            "message": f"Failed to send alert: {str(e)}",
            "subscribers_count": len(subscribers)
        }

@telegram_api.get("/status")
async def api_get_status():
    """
    Get Telegram bot status and statistics
    
    Returns:
        dict: Bot status information
    """
    return {
        "status": "running" if application else "stopped",
        "subscribers_count": len(subscribers),
        "last_alert": get_last_fire_alert(),
        "bot_running": is_bot_running()
    }

@telegram_api.get("/subscribers")
async def api_get_subscribers():
    """
    Get current subscribers list
    
    Returns:
        dict: Subscribers information
    """
    return {
        "subscribers_count": len(subscribers),
        "subscribers": list(subscribers)
    }

def start_telegram_api():
    """Start the Telegram API server in a separate thread"""
    def run_api():
        uvicorn.run(telegram_api, host="0.0.0.0", port=5002, log_level="info")
    
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    print("🌐 Telegram API server started on http://localhost:5002")
    print("Available API endpoints:")
    print("  POST /send-fire-alert - Send fire alert with image")
    print("  POST /send-fire-alert-with-risk - Send fire alert with risk level")
    print("  POST /send-simple-alert - Send simple text alert")
    print("  GET  /status - Get bot status")
    print("  GET  /subscribers - Get subscribers info")

# ============ END HTTP API ============

if __name__ == '__main__':
    # Start the HTTP API server first
    start_telegram_api()
    
    # Wait a moment for the API to start
    time.sleep(2)
    
    # Initialize and start the Telegram bot
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('last_fire', last_fire))
    application.add_handler(CommandHandler('set_alert', set_alert))
    #application.add_handler(CommandHandler('secret', secret))

    print("🔥 Fire Prediction Telegram Bot is running...")
    print("Available commands:")
    print("  /start - Welcome message")
    print("  /last_fire - Show last fire analysis with image")
    print("  /set_alert - Subscribe to fire alerts")
    print("\nExternal API functions available:")
    print("  - send_fire_alert(image_path, message)")
    print("  - send_custom_fire_alert(image_path, custom_message, job_id=None)")
    print("  - update_last_fire_alert(image_path, message)")
    print("  - send_fire_alert_with_risk_level(image_path, job_id, risk_level, custom_message=None)")
    print("  - send_simple_alert(message_text)")
    print("  - get_subscriber_count()")
    print("  - get_last_fire_alert()")
    print("\nHTTP API running on http://localhost:5002")
    
    application.run_polling()
