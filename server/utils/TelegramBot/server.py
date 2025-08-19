import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
import threading
import time
import asyncio

TELEGRAM_TOKEN = "8365782927:AAGqQ2M-uTs_LcMSbo8E4gW387_EFkF1uHE"

# Configure logging to see errors
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# List of users subscribed to alerts
subscribers = set()
application = None

# Global variable for fire analysis message
current_fire_message = "Latest fire analysis: No risks currently detected."

# /start command: tells you to use /last_fires
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Welcome! Use /last_fires to see the latest fire analysis. If you want to receive alerts, use /set_alert."
    )

# /last_fires command: shows the latest fire analysis
async def last_fires(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=current_fire_message
    )

# /set_alert command: subscribes the user to alerts
async def set_alert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    subscribers.add(user_id)
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="You have subscribed to fire risk alerts."
    )

# ============ EXTERNAL API FUNCTIONS ============

def update_fire_message(new_message):
    """
    Update the fire analysis message that will be shown with /last_fires command
    
    Args:
        new_message (str): The new fire analysis message
    """
    global current_fire_message
    current_fire_message = new_message
    logging.info(f"Fire message updated: {new_message}")

def send_alert_to_subscribers(message):
    """
    Send a custom alert message to all subscribed users
    
    Args:
        message (str): The alert message to send
    """
    if subscribers and application:
        for user_id in subscribers:
            try:
                # Create new event loop for this thread if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Send message
                loop.run_until_complete(
                    application.bot.send_message(chat_id=user_id, text=message)
                )
            except Exception as e:
                logging.error(f"Error sending alert to {user_id}: {e}")
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

# ============ END EXTERNAL API FUNCTIONS ============

# Send alert to subscribers (synchronous function)
def send_fire_alert():
    """Internal function for periodic fire alerts"""
    send_alert_to_subscribers("New fire risk detected!")

# Periodic task in separate thread
def periodic_fire_alerts():
    while True:
        time.sleep(30)  # Wait 60 seconds
        send_fire_alert()

if __name__ == '__main__':
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('last_fires', last_fires))
    application.add_handler(CommandHandler('set_alert', set_alert))

    # Start the periodic task in a separate thread
    alert_thread = threading.Thread(target=periodic_fire_alerts, daemon=True)
    alert_thread.start()

    print("The bot is running...")
    
    # Mock functionality for testing
    def mock_fire_updates():
        """Mock function to simulate fire updates every 2 minutes"""
        import random
        messages = [
            "Fire risk level: LOW - No immediate threats detected",
            "Fire risk level: MEDIUM - Dry conditions in some areas", 
            "Fire risk level: HIGH - Strong winds and dry vegetation detected",
            "Fire risk level: CRITICAL - Active fire spotted, authorities notified"
        ]
        
        while True:
            time.sleep(10)  # Wait 2 minutes
            random_message = random.choice(messages)
            update_fire_message(random_message)
            print(f"Mock update: {random_message}")
            
            # Simulate alert for high/critical levels
            if "HIGH" in random_message or "CRITICAL" in random_message:
                alert_msg = f"⚠️ ALERT: {random_message}"
                send_alert_to_subscribers(alert_msg)
                print(f"Mock alert sent: {alert_msg}")
    
    # Start mock updates in a separate thread
    mock_thread = threading.Thread(target=mock_fire_updates, daemon=True)
    mock_thread.start()
    print("Mock fire monitoring started (updates every 2 minutes)")
    
    application.run_polling()
