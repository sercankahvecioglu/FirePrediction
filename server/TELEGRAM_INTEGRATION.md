# Fire Prediction System - Telegram Integration

This system integrates Telegram bot notifications with the fire prediction satellite analysis API.

## Architecture

The system consists of two independent services that communicate via HTTP API:

1. **Main FastAPI Server** (`app.py`) - Port 5001
   - Processes satellite images
   - Performs fire risk analysis
   - Sends alerts to Telegram bot via HTTP requests

2. **Telegram Bot Server** (`utils/TelegramBot/server.py`) - Port 5002
   - Handles Telegram bot interactions
   - Receives alert requests via HTTP API
   - Manages subscriber notifications

## Quick Start

### Option 1: Start Both Services Together (Recommended)

```bash
cd server
python start_services.py
```

This will start both services and monitor them together.

### Option 2: Start Services Manually

**Terminal 1 - Start Telegram Bot:**
```bash
cd server/utils/TelegramBot
python server.py
```

**Terminal 2 - Start Main API:**
```bash
cd server
python app.py
```

## API Endpoints

### Main API (Port 5001)
- `GET /` - Health check with Telegram status
- `POST /test-telegram-alert` - Test Telegram integration
- `GET /telegram-status` - Check Telegram bot connection
- `POST /submit-image/fire-prediction` - Submit image for fire analysis

### Telegram Bot API (Port 5002)
- `POST /send-fire-alert` - Send fire alert with image
- `POST /send-fire-alert-with-risk` - Send alert with risk level
- `POST /send-simple-alert` - Send text-only alert
- `GET /status` - Get bot status and statistics

## Testing the Integration

1. **Start both services**
2. **Test the connection:**
   ```bash
   curl http://localhost:5001/telegram-status
   ```

3. **Send a test alert:**
   ```bash
   curl -X POST "http://localhost:5001/test-telegram-alert?message=Hello%20from%20API"
   ```

4. **Chat with the Telegram bot:**
   - Start a chat with your bot
   - Use `/start` to get welcome message
   - Use `/set_alert` to subscribe to notifications
   - Use `/last_fire` to see the latest analysis

## Telegram Bot Commands

- `/start` - Welcome message and instructions
- `/set_alert` - Subscribe to fire risk notifications
- `/last_fire` - View latest fire risk analysis with heatmap

## Configuration

### Telegram Bot Token
Update the `TELEGRAM_TOKEN` in `utils/TelegramBot/server.py`:
```python
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
```

### API URLs
If running on different hosts/ports, update `TELEGRAM_BOT_API_URL` in `app.py`:
```python
TELEGRAM_BOT_API_URL = "http://localhost:5002"
```

## Fire Risk Analysis Flow

1. **Image Upload** → Main API receives satellite image
2. **Processing** → Cloud detection, forest detection, fire prediction
3. **Risk Assessment** → Analyze fire probability and risk level
4. **Alert Generation** → Create risk message and heatmap
5. **Telegram Notification** → Send HTTP request to Telegram bot
6. **User Notification** → Bot sends alerts to subscribed users

## Risk Levels

- 🟢 **LOW** - No immediate threats detected
- 🟡 **MEDIUM** - Some elevated conditions detected
- 🟠 **HIGH** - Significant fire risk detected
- 🔴 **CRITICAL** - Extreme fire danger

## Troubleshooting

### Telegram Bot Not Responding
1. Check if bot is running: `curl http://localhost:5002/status`
2. Verify bot token is correct
3. Ensure port 5002 is not blocked

### No Alerts Received
1. Subscribe to alerts with `/set_alert` command
2. Check bot status: `GET /telegram-status`
3. Test with: `POST /test-telegram-alert`

### Connection Issues
1. Verify both services are running
2. Check firewall settings for ports 5001 and 5002
3. Ensure services can communicate on localhost

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `fastapi` - Web framework
- `python-telegram-bot` - Telegram bot library
- `requests` - HTTP client for inter-service communication
- `uvicorn` - ASGI server

## Production Deployment

For production deployment:

1. **Use environment variables** for sensitive data
2. **Deploy services separately** with proper load balancing
3. **Use Redis or database** for persistent storage
4. **Implement proper logging** and monitoring
5. **Set up SSL/TLS** for secure communication

## Support

For issues or questions:
1. Check the logs from both services
2. Test individual components separately
3. Verify Telegram bot permissions and token
