# Plant Pathology Backend — Setup Guide

## 1. Plug in your trained model

- Copy your saved model file into this folder and rename it to
  `plant_disease_model.h5` (or update `MODEL_PATH` in `app.py`).
- Open `app.py` and fill in:
  - `IMAGE_SIZE` — must match what you trained with
  - `CLASS_NAMES` — the exact list and order of your output classes
    (check `train_generator.class_indices` from your training notebook)
  - The normalization line in `preprocess_image()` — must match your
    training preprocessing exactly, or predictions will be wrong

## 2. Run it locally

```bash
cd plant-backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Test it:
```bash
curl -X POST -F "file=@test_leaf.jpg" http://localhost:5000/predict
```

## 3. Deploy it (so your AI Studio frontend can reach it)

Pick one — all have free tiers suitable for a demo:

**Render** (easiest)
1. Push this folder to a GitHub repo
2. On render.com → New → Web Service → connect the repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Render gives you a URL like `https://your-app.onrender.com`

**Railway** — similar flow, auto-detects Flask apps from the repo.

**Hugging Face Spaces** (Docker SDK) — good if your model file is large,
since HF handles large file storage well via Git LFS.

⚠️ Note: your model file (`.h5`) can be large. If it's over ~100MB, use
Git LFS or the platform's large-file upload option, since regular git
pushes will fail.

## 4. Connect it to your AI Studio frontend

In your frontend's `predictDisease(imageFile)` function, replace the mock
with:

```javascript
async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("https://your-app.onrender.com/predict", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (data.result === "rejected") {
    // show the "not a recognizable leaf" UI state
    return { rejected: true, message: data.message };
  }

  return {
    disease: data.disease,
    confidence: data.confidence,
    description: data.description,
  };
}
```

Set `USE_MOCK = false` once this is working.

## 5. Common issues

- **CORS errors in browser console** → confirm `flask-cors` is installed
  and `CORS(app)` is present in `app.py`.
- **Predictions look random/wrong** → almost always a preprocessing
  mismatch between training and inference (wrong image size, wrong
  normalization, or wrong class order).
- **Model file too big to deploy** → consider converting to TensorFlow
  Lite (`.tflite`) for a smaller footprint, or host the model separately
  (e.g. Hugging Face Model Hub) and load it at startup.
- **Cold start is slow** → free tiers on Render/Railway spin down when
  idle; the first request after inactivity can take 30–60s to load the
  model.
