# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY public/ ./public/

COPY . .

ENV VITE_API_URL=0.0.0.0
ENV VITE_API_PORT=8000

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start", "--host"]