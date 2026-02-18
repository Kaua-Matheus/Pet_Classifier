# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY public/ ./public/

COPY . .

ENV VITE_API_URL="IP público"
ENV VITE_API_PORT=443

RUN npm run build

EXPOSE 80

CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "80"]