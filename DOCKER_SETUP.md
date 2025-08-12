# Docker Setup

This guide explains how to set up and run the FirePrediction web application using Docker.

## Prerequisites

Before building or running anything in the `/server` directory, ensure that a `/secrets` folder exists inside `/server`. This folder must contain the following files:
- `token.json`
- `client-secret.json`

These files are required for authentication and secure access.

## Docker Compose Configuration

The application uses Docker Compose with the following services:

### 1. Frontend

- **Build Context:** `./client`
- **Dockerfile:** `Dockerfile`
- **Container Name:** `flamesentinels-frontend`
- **Ports:** Maps port `80` in the container to `3000` on your host (access via `localhost:3000`)
- **Depends On:** Backend service
- **Network:** `flamesentinels-network`

### 2. Backend

- **Build Context:** `./server`
- **Dockerfile:** `Dockerfile`
- **Container Name:** `flamesentinels-backend`
- **Ports:** Maps port `5000` in the container to `5001` on your host (for debugging)
- **Network:** `flamesentinels-network`

### 3. Network

Both services communicate over a custom bridge network named `flamesentinels-network`.

## Usage

1. Place `token.json` and `client-secret.json` inside `/server/secrets`.
2. Run the following command in the project root:

     ```bash
     docker-compose up --build
     ```

3. Access the frontend at [http://localhost:3000](http://localhost:3000).
