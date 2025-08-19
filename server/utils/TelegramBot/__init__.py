"""
Telegram Fire Alert Bot

A Telegram bot for fire risk monitoring and alerts.

Available functions:
- update_fire_message(new_message): Update the fire analysis message
- send_alert_to_subscribers(message): Send custom alerts to subscribers
- get_subscribers(): Get list of subscribed users
- add_subscriber(user_id): Add a user to alerts
- remove_subscriber(user_id): Remove a user from alerts
- is_bot_running(): Check if bot is running
"""

from .server import (
    update_fire_message,
    send_alert_to_subscribers,
    get_subscribers,
    add_subscriber,
    remove_subscriber,
    is_bot_running
)

__version__ = "1.0.0"
__author__ = "Fire Alert Bot Team"

__all__ = [
    "update_fire_message",
    "send_alert_to_subscribers", 
    "get_subscribers",
    "add_subscriber",
    "remove_subscriber",
    "is_bot_running"
]
