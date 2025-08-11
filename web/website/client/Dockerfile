# Step 1: Build React app
FROM node:18 AS build
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code and build
COPY . .
RUN npm run build

# Step 2: Serve with Nginx
FROM nginx:1.25
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config (this handles /api proxy to backend)
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
