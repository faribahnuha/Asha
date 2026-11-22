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

## Model (v2: chance of success)

`POST /recommend` now ranks hospitals by an estimated **chance of success**:

> P(success) = P(capable) × P(right blood products available in time) × P(patient arrives within the clinical window)

- **P(capable)**: a logistic-regression model trained on 240 reviewed labels (`labels/`), using evidence
  from each facility's own record; raised where LaQshya independently certifies obstetric surgery.
  Out-of-fold AUC 0.94 vs 0.80 / 0.73 for simple baselines (`modeling/capability/evaluation_report.md`).
- **P(blood in time)**: the hospital's own blood bank (licensed in the government directory, or only
  claimed in its listing) plus the nearest licensed banks that stock the scenario's products, with
  transport and issue time, night-time hours and Rh-negative stock.
- **P(arrives in time)**: straight-line distance × detour / speed (urban or rural from local density),
  with a travel-time spread, against the scenario's window (shorter if Hb is known to be low).
- Every assumption is in `dataset/ref/parameters.csv`; every distance is computed from coordinates
  at query time. Below 50% the API returns `status: "no_confident_option"` with a message.

Each result keeps the fields the UI already used and adds `p_success`, the three factors, verbatim
`evidence` quotes for each required capability, `blood_source`, `call_ahead` questions and
`location_confidence`. Optional request fields: `hb` (g/dL), `rh_negative`, `night`.

Data pipeline: `pre-processing/` (bronze → silver → gold, LaQshya parser). Gold tables are in
`dataset/gold/`. The previous entropy-TOPSIS API is kept as `modeling/api/app_v1_topsis.py`.

Run the checks: `python3 modeling/self_checks/run_all_checks.py`.

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
