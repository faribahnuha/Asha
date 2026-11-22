# Asha care finder

A responsive React UI connected to the repository's real Python facility recommendation model.

## Run locally

Terminal 1 (model API):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r modeling/api/requirements.txt
uvicorn modeling.api.app:app --reload --port 8000
```

Terminal 2 (web app):

```bash
npm install
npm run dev
```

Open `http://localhost:5173`.

## Deployment

Deploy the Python API using the included `render.yaml`. Then deploy this repository to Vercel and add:

```text
VITE_ASHA_API_URL=https://your-service-name.onrender.com
```

Redeploy after adding or changing the variable. No secret API key is required.

## Safety and data limitations

- Results are model estimates, not clinical advice or live routing.
- The map uses the actual patient and model-facility latitude/longitude values. OpenStreetMap supplies only the background tiles.
- Typed Indian place names are converted to coordinates; browser geolocation can be used instead.
- Location suggestions appear after at least three characters; selecting one stores its real coordinates before running the model.
- Current-location access requires browser permission and a secure origin (`https://` or `localhost`).
- The interface supports English, Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, and Urdu.
- Travel time is estimated from straight-line distance with fixed assumptions.
- Service data is incomplete; the interface exposes unconfirmed facilities and asks users to call ahead.
- For immediate danger, call the appropriate emergency service (112 in India).
