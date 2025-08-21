# Docker Deployment Guide

This guide explains how to deploy the Fire Prediction System using Docker.

## Quick Start

### 1. Using the Management Script (Recommended)

```bash
# Make the script executable (first time only)
chmod +x docker-manage.sh

# Build and start the services
./docker-manage.sh build
./docker-manage.sh start

# Check status
./docker-manage.sh status

# View logs
./docker-manage.sh logs

# Test the services
./docker-manage.sh test
```

### 2. Using Docker Compose Directly

```bash
# Build the image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

The Docker setup runs both services in a single container:

- **Main API Server**: Port 5001
- **Telegram Bot API**: Port 5002

## Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:

```env
TELEGRAM_TOKEN=your_bot_token_here
MAIN_API_PORT=5001
TELEGRAM_API_PORT=5002
ENVIRONMENT=production
```

## File Structure

```
FirePrediction/
├── docker-compose.yml          # Docker Compose configuration
├── nginx.conf                  # Nginx reverse proxy config
├── docker-manage.sh           # Management script
├── .env.example              # Environment variables template
└── server/
    ├── Dockerfile            # Docker image definition
    ├── requirements.txt      # Python dependencies
    ├── start_services.py    # Service startup script
    └── ...
```

## Management Script Commands

| Command | Description |
|---------|-------------|
| `build` | Build the Docker image |
| `start` | Start the services |
| `stop` | Stop the services |
| `restart` | Restart the services |
| `logs` | Show service logs |
| `status` | Show service status |
| `clean` | Clean up containers and images |
| `test` | Test the services |
| `shell` | Access the container shell |
| `help` | Show help message |

## Testing

### Health Checks

```bash
# Check main API
curl http://localhost:5001/

# Check Telegram API
curl http://localhost:5002/status

# Test Telegram integration
curl -X POST "http://localhost:5001/test-telegram-alert?message=Test"
```

### API Documentation

Once running, access the interactive API documentation:
- **Main API**: http://localhost:5001/docs
- **Telegram API**: http://localhost:5002/docs

## Production Deployment

### With Nginx Reverse Proxy

```bash
# Start with nginx reverse proxy
docker-compose --profile production up -d
```

This will:
- Start the backend services
- Run nginx on ports 80/443
- Proxy requests to appropriate services

### SSL Configuration

1. Obtain SSL certificates
2. Place them in `./ssl/` directory
3. Uncomment SSL configuration in `nginx.conf`
4. Restart services

## Volumes and Persistence

The container persists data in:

- `./server/data/` - Processing data and temporary files
- `./results/` - Generated fire analysis results

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose logs backend

# Access container shell
./docker-manage.sh shell

# Check system resources
docker system df
```

### Services Not Responding

```bash
# Check service status
./docker-manage.sh status

# Restart services
./docker-manage.sh restart

# View real-time logs
./docker-manage.sh logs
```

### Port Conflicts

If ports 5001 or 5002 are in use:

1. Edit `docker-compose.yml`
2. Change port mappings:
   ```yaml
   ports:
     - "8001:5001"  # Use port 8001 instead of 5001
     - "8002:5002"  # Use port 8002 instead of 5002
   ```

### Memory Issues

```bash
# Check container resource usage
docker stats

# Increase Docker memory limits if needed
# (Docker Desktop: Settings > Resources > Memory)
```

## Development vs Production

### Development Mode

```bash
# Use development settings
./docker-manage.sh start
```

Features:
- Auto-reload enabled
- Debug logging
- Development ports exposed

### Production Mode

```bash
# Use production profile with nginx
docker-compose --profile production up -d
```

Features:
- Nginx reverse proxy
- SSL termination
- Production logging
- Health checks

## Monitoring

### Container Health

```bash
# Check container health
docker-compose ps

# View health check logs
docker inspect flamesentinels-backend --format='{{.State.Health}}'
```

### Application Metrics

Access metrics endpoints:
- Main API metrics: `GET /status`
- Telegram bot metrics: `GET /telegram-status`

## Backup and Recovery

### Data Backup

```bash
# Backup data directory
tar -czf backup-$(date +%Y%m%d).tar.gz server/data/ results/

# Backup Docker image
docker save flamesentinels-backend > flamesentinels-backup.tar
```

### Recovery

```bash
# Restore data
tar -xzf backup-YYYYMMDD.tar.gz

# Load Docker image
docker load < flamesentinels-backup.tar
```

## Security Considerations

1. **Change default Telegram token** in production
2. **Use environment variables** for sensitive data
3. **Enable SSL/TLS** for public deployment
4. **Restrict network access** using Docker networks
5. **Regular security updates** of base images

## Performance Tuning

### Resource Limits

Add to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'
```

### Scaling

For high-load scenarios:

```bash
# Scale backend service
docker-compose up -d --scale backend=3
```

## Support

For issues:
1. Check logs: `./docker-manage.sh logs`
2. Test services: `./docker-manage.sh test`
3. Access shell: `./docker-manage.sh shell`
4. Review this documentation
