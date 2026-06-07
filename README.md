# 🌿 Plant Disease Predictor — Docker + Streamlit Setup

## Project Structure

```
plant_disease_app/
├── app.py                              ← Streamlit application
├── requirements.txt                    ← Python dependencies
├── Dockerfile                          ← Docker image definition
├── docker-compose.yml                  ← Easy start/stop commands
├── .dockerignore                       ← Files excluded from image
├── plant_disease_prediction_model.h5   ← ⬅ YOU MUST ADD THIS
└── class_indices.json                  ← ⬅ YOU MUST ADD THIS
```

---

## Step-by-Step Instructions

### Step 1 — Export Files from Colab

Run these two cells at the end of your Colab notebook:

```python
# Save the model
model.save('plant_disease_prediction_model.h5')

# Save class indices
import json
json.dump(class_indices, open('class_indices.json', 'w'))
```

Then download both files from the Colab Files panel (📁 icon on the left sidebar).

---

### Step 2 — Set Up Local Folder

Create a folder and place all project files inside it:

```
plant_disease_app/
├── app.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── plant_disease_prediction_model.h5   ← paste downloaded model here
└── class_indices.json                  ← paste downloaded json here
```

---

### Step 3 — Install Docker

If Docker is not installed:

- **Windows / Mac**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Ubuntu/Linux**:
  ```bash
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose
  sudo systemctl start docker
  sudo usermod -aG docker $USER   # re-login after this
  ```

---

### Step 4 — Build the Docker Image

Open a terminal inside your `plant_disease_app/` folder and run:

```bash
docker build -t plant-disease-predictor .
```

You will see TensorFlow and all dependencies being installed.
This takes **3–7 minutes** on first build (cached on subsequent builds).

---

### Step 5 — Run the Container

#### Option A — Simple `docker run` command:
```bash
docker run -p 8501:8501 plant-disease-predictor
```

#### Option B — Using Docker Compose (recommended):
```bash
docker-compose up
```

To run in background (detached mode):
```bash
docker-compose up -d
```

---

### Step 6 — Open the App

Open your browser and go to:

```
http://localhost:8501
```

Upload any plant leaf image and click **Predict Disease**!

---

## Useful Docker Commands

| Command | Description |
|---|---|
| `docker-compose up` | Start the app |
| `docker-compose up -d` | Start in background |
| `docker-compose down` | Stop the app |
| `docker-compose logs -f` | View live logs |
| `docker ps` | List running containers |
| `docker images` | List built images |
| `docker-compose build --no-cache` | Force rebuild from scratch |

---

## Stopping the App

```bash
# If using docker run → press Ctrl+C in terminal

# If using docker-compose
docker-compose down
```

---

## Troubleshooting

### Model not found error
Make sure `plant_disease_prediction_model.h5` is in the same folder as `Dockerfile`.

### Port already in use
Change the host port in `docker-compose.yml`:
```yaml
ports:
  - "8502:8501"   # use 8502 instead
```
Then open `http://localhost:8502`.

### Out of memory during build
TensorFlow is large. Ensure Docker has at least **4 GB RAM** allocated:
- Docker Desktop → Settings → Resources → Memory → set to 4 GB+

### Wrong predictions / class names missing
Ensure `class_indices.json` was generated from the **same training run** as the model.
