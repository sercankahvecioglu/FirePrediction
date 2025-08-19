# Telegram Fire Alert Bot

A Telegram bot for fire risk monitoring and alerts with external API support.

## Features

- `/start` - Welcome message and instructions
- `/last_fires` - Show latest fire analysis
- `/set_alert` - Subscribe to fire risk alerts
- External API for integration with other systems

## Installation

```bash
pip install python-telegram-bot
```

## Usage

### Running the bot directly
```bash
python server.py
```

### Using as a module
```python
from TelegramBot import update_fire_message, send_alert_to_subscribers

# Update fire analysis
update_fire_message("High fire risk in northern regions!")

# Send custom alert
send_alert_to_subscribers("Emergency evacuation required!")
```

## API Functions

- `update_fire_message(message)` - Update the fire analysis message
- `send_alert_to_subscribers(message)` - Send alerts to all subscribers
- `get_subscribers()` - Get list of subscribed users
- `add_subscriber(user_id)` - Add user to alerts
- `remove_subscriber(user_id)` - Remove user from alerts
- `is_bot_running()` - Check if bot is running

## Mock Mode

When running directly, the bot includes mock functionality that:
- Updates fire messages every 2 minutes with random risk levels
- Sends alerts for HIGH and CRITICAL risk levels
- Simulates real fire monitoring system behavior
