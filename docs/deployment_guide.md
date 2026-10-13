# Deployment Guide: AI Book Recommendation System

This guide outlines all verified deployment strategies for the **AI Book Recommendation System**, ranging from instant zero-setup local deployment to 100% free production cloud hosting.

---

## 🚀 Deployment Methods Overview

| Deployment Method | Cost | Effort | Best For |
| :--- | :--- | :--- | :--- |
| **1. Local Network / LAN** | Free | Instant (Running Now) | College viva demo on student laptop / phone |
| **2. Streamlit Community Cloud** | Free | 2 minutes (GitHub push) | Permanent public URL for project submission |
| **3. Hugging Face Spaces** | Free | 3 minutes | Academic AI portfolio & public demo |
| **4. Docker Container** | Free | 1 command | Local servers, edge devices, enterprise cloud |
| **5. Render / Railway** | Free / Low | 5 minutes | Cloud web service hosting |

---

## 1. Local Network / LAN Deployment (Active Now)

The application is already running on your machine:
- **Localhost URL:** `http://localhost:8501`
- **Network / LAN URL:** `http://10.121.243.113:8501`

### Accessing from Mobile Phones or Other Laptops on Same Wi-Fi:
1. Connect any other device (phone, examiner's laptop, tablet) to the same Wi-Fi network.
2. Open the mobile browser and visit:  
   `http://10.121.243.113:8501`
3. The responsive Streamlit interface will render smoothly with full interactivity.

### Windows One-Click Launch:
Double-click `run_app.bat` in the project root directory anytime you want to launch the system.

---

## 2. 100% Free Cloud Deployment: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is free forever, requires zero credit card, and provides a custom URL like `https://ai-book-recommender.streamlit.app`.

### Steps:
1. **Initialize Git & Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit of AI Book Recommendation System"
   git branch -M main
   # Create a public or private repository on GitHub, then link it:
   git remote add origin https://github.com/<your-username>/ai-book-recommendation-system.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **"New app"**.
   - Select your repository (`<your-username>/ai-book-recommendation-system`).
   - Set Main file path: `app.py`.
   - Click **"Deploy!"**.
   - In ~60 seconds, your application is live on the internet!

---

## 3. 100% Free Cloud Deployment: Hugging Face Spaces

Hugging Face Spaces natively supports Streamlit apps with free CPU hardware.

### Steps:
1. Create an account at [huggingface.co](https://huggingface.co).
2. Go to **Spaces** -> **Create new Space**.
3. Choose **Space SDK: Streamlit**.
4. Set Space name (e.g., `ai-book-recommender`).
5. Push your project files or link your GitHub repository.
6. The app will build and serve instantly.

---

## 4. Docker Container Deployment

The repository includes a ready-to-run multi-stage `Dockerfile` and `docker-compose.yml`.

### Prerequisites:
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (if deploying on a machine with Docker).

### Build and Run with Docker Compose:
```bash
docker compose up -d --build
```

### Access:
Visit `http://localhost:8501`.

### Useful Docker Commands:
```bash
# Check container status
docker ps

# View application logs
docker logs -f ai_book_recommender

# Stop the container
docker compose down
```

---

## 5. Environment Variables & Port Configuration

The server configuration in `.streamlit/config.toml` automatically configures:
- `server.port = 8501`
- `server.address = "0.0.0.0"`
- `server.headless = true`
- `browser.gatherUsageStats = false`

To change the port dynamically:
```bash
streamlit run app.py --server.port 8080
```
