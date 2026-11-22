# Deploying This as an Actual App

This covers three separate things people tend to conflate: running the model itself, the UI
drafts that exist so far, and putting a live, callable version of this online for free. Each
section below was actually run, not just written — see "verified" notes inline.

## 1. Running the model (`modeling/`)

Everything here is plain Python + pandas/numpy, no training step, no GPU, no API keys. Clone the
repo, then from its root:

```bash
# Confirm the model logic itself is correct (synthetic data, runs in under a second)
python3 modeling/self_checks/run_all_checks.py

# Try it on a real location from the real dataset
python3 modeling/demo/try_it_out.py --scenario injury_bleeding --lat 19.07 --lon 72.88

# Or let it drop you at a random real place in India
python3 modeling/demo/try_it_out.py --scenario pregnancy_bleeding --seed 42

# See how much the model's other criteria (vs. plain distance) actually change the answer
python3 modeling/demo/compare_distance_only.py
```

`try_it_out.py` is the one to point someone at if they just want to see the model work without
touching HTTP or a browser.

## 2. The UI drafts

**Facility Match Review** (the dashboard): https://claude.ai/artifact/GiegZnNvCD1JmG1kYxzJ37

This is a **static draft**, not the production UI — it's six real locations run through the real
model once, with the results baked into the page. There's no live computation happening when you
click between cases; it's there to review what the model's output *looks like* (position plot,
confirmed/unconfirmed capability chips, drive time, scoring weights) before anyone commits to
building a real interactive frontend around it. If the UI direction in that draft is approved, a
real frontend (Section 3) would call the API below instead of reading baked-in JSON.

To regenerate that draft's data after a model change:

```bash
python3 modeling/demo/generate_dashboard_data.py   # writes modeling/demo/dashboard_data.json
```

(The JSON then needs to be re-embedded into `modeling/demo/dashboard.html` and republished — ask
whoever has access to the artifact to do that, since it's a separate manual step, not automated.)

## 3. Putting a live API online, for free

### What's already built
`modeling/api/app.py` is a FastAPI wrapper around the exact same `recommend()` function the CLI
demo calls — nothing is reimplemented. Two endpoints:

- `GET /health` — confirms the process is up and the dataset loaded
- `POST /recommend` — body `{"lat": ..., "lon": ..., "scenario": "..."}`, returns the top-3 JSON

Tested locally (both via FastAPI's TestClient and a real running `uvicorn` process with `curl`) —
including the padding-to-3 and unconfirmed-fallback paths. CORS is already enabled (`allow_origins:
["*"]`) so a frontend on a different domain can call it without the browser silently blocking the
request — verified with a real preflight `OPTIONS` call.

### Why Render, specifically (not Vercel)

This was a real back-and-forth, not an arbitrary pick:

- **Vercel is Node/JS-first.** Its Python support is a secondary runtime, and pandas + numpy are
  large, compiled-binary-heavy packages — a known source of friction there.
- **Vercel functions are stateless and short-lived.** The thing that makes this API fast — loading
  the ~21MB of CSVs *once* at process startup instead of per-request — doesn't reliably survive
  between invocations in that model. Every cold start would likely mean re-reading both CSVs from
  scratch.
- **Render runs Python as a normal, persistent process** (like any traditional server), so "load
  once at startup" actually behaves the way the code expects.

Verified directly against Render's own current docs (not assumed, since pricing/limits pages
change): the free tier gives **750 instance-hours/month** (enough to run continuously, all month,
for free), **no credit card required to start**. The one real trade-off: a free service **spins
down after 15 minutes with no traffic** and takes about a minute to wake up on the next request.
For a demo or low-traffic use, that's a fine trade for $0/month; it's not something you'd accept
for a production app with real users expecting instant responses.

If this ever needs to be always-warm, that's the point to revisit the hosting choice — not before.

### Steps to actually deploy it

1. Go to [render.com](https://render.com) and sign up (no card needed for this).
2. **New → Blueprint**, connect this GitHub repo.
3. Render reads `render.yaml` (already in this repo's root) and deploys `modeling/api/app.py`
   automatically — no manual build/start command typing.
4. You'll get a URL like `https://asha-recommend-api.onrender.com`.
5. Confirm it's alive: `curl https://asha-recommend-api.onrender.com/health`

If you're demoing this live, hit `/health` a minute or two before you present — that wakes a
spun-down service up before anyone's watching.

### Wiring an actual frontend to it

Whatever hosts the UI (Vercel, Netlify, GitHub Pages, whatever) calls the Render URL like any
other API:

```js
const response = await fetch("https://asha-recommend-api.onrender.com/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ lat: 19.0760, lon: 72.8777, scenario: "injury_bleeding" }),
});
const result = await response.json();
```

That's it — the frontend and the model backend are two separate, independently-hosted things
talking over plain HTTP. This is the normal shape for pairing a JS frontend with a Python
data/ML backend; they essentially never share one runtime.

## Known limitations that carry into any deployment

These are already documented in `model-design.md (draft)` and `modeling/CLAUDE.md`, restated here
because they matter just as much once this is "a real app," not just a script:

- **Distance is straight-line, not routed** (no OSM/live routing, by deliberate project decision —
  see `candidate_pool.py`'s header). `detour_factor` and `avg_speed_kmph` are **assumed, not
  measured** placeholder values (1.3x, 35 km/h) — every API response should probably surface that
  assumption to an end user, not just bury it in code comments, before this is treated as
  authoritative for real routing decisions.
- **Most candidates (~85% of the pool) have no confirmed service-capability data at all.** The
  model handles this honestly (a confirmed/unconfirmed split, never guessing a number for missing
  data — see `model-design.md`'s "Unknown vs. confirmed-absent" section), but a real frontend needs
  to surface the "capability not confirmed — call ahead" note just as prominently as the ranking
  itself, not as a small-print afterthought.
- **This has not been through the critical-reviewer pass** the project's own root `CLAUDE.md`
  calls for before a demo or submission — that agent file still doesn't exist in this repo as of
  this writing.
