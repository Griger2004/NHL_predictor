FROM node:22-alpine
WORKDIR /app
COPY main/frontend/package*.json ./
RUN npm install
COPY main/frontend/ .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
