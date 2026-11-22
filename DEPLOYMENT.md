# Publish Asha

The app has two deployments: the Python recommendation API on Render and the React interface on Vercel. There are no API keys.

## 1. Put the project on GitHub

Create an empty GitHub repository, unzip this project, open its folder in Terminal, and run:

```bash
git init
git add .
git commit -m "Build Asha care finder"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/asha.git
git push -u origin main
```

## 2. Deploy the model API on Render

1. Sign in at https://render.com with GitHub.
2. Choose **New + → Blueprint**.
3. Select the GitHub repository and approve the included `render.yaml`.
4. Wait for the service to become live.
5. Copy its public URL, such as `https://asha-recommend-api.onrender.com`.
6. Open `YOUR-URL/health`. It should show `{"status":"ok", ...}`.

The free Render tier can sleep while unused, so the first request may take a while. Open `/health` shortly before a demo.

## 3. Deploy the UI on Vercel

1. Sign in at https://vercel.com with GitHub.
2. Choose **Add New → Project** and import the same repository.
3. Vercel should detect **Vite**. Keep the defaults:
   - Build command: `npm run build`
   - Output directory: `dist`
4. Expand **Environment Variables** and add:
   - Name: `VITE_ASHA_API_URL`
   - Value: the Render URL from step 2, with no trailing slash
5. Click **Deploy**.

If you add the variable after the first deployment, open **Deployments**, use the menu on the latest deployment, and select **Redeploy**.

## 4. Verify the live app

Open the Vercel URL and check all three care types. Try **Use current location**, allow browser permission, then confirm that selecting a result opens the hospital details sheet.

Type another Indian town or district and confirm the map recenters. The numbered hospital markers use the exact coordinates returned by the model. Test all eight choices in the language menu.

When testing **Use current location**, allow the location permission prompt. If it was previously denied, re-enable Location for the site in the browser's site settings and reload the page.

## 5. Optional: deploy as a Databricks App

A second deployment of the same frontend + API as a single Databricks App lives in `databricks_app/` (`server.py` mounts the built `dist/` on top of the unchanged `modeling/api/app.py`). Steps:

```bash
npm install
VITE_ASHA_API_URL="" npm run build   # relative /recommend calls, same-origin
databricks apps create asha-care-finder
databricks workspace mkdirs /Workspace/Users/<you>/apps/asha-care-finder
databricks workspace import-dir . /Workspace/Users/<you>/apps/asha-care-finder --overwrite
databricks apps deploy asha-care-finder \
  --source-code-path /Workspace/Users/<you>/apps/asha-care-finder
```

**Important limitation of this deployment path**: Databricks Apps has no public/anonymous access mode. Every viewer must have a Databricks account and be explicitly granted `CAN_USE` on the app (`databricks apps set-permissions`), which real ambulance drivers in the field will not have. This path exists to demonstrate Databricks usage (e.g. for a sponsor track); the Vercel deployment above is the one meant for actual field use, since it is a public URL with no login.

## Important limitations

- The route and travel time are estimates, not live navigation.
- Facility service records can be incomplete, so the app labels unconfirmed capability and asks users to call ahead.
- This is a decision-support prototype, not medical advice or a substitute for emergency services.
