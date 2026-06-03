# UOT Temporal Extrapolation Engine — Deployment Checklist

## Files needed in your repository
```
uot_api_v2.py               ← backend (required)
uot_engine_v12_patched.py   ← engine (required)
uot_live_search.py          ← live search module (required for LIVE_MODE=true)
uot_frontend_v2.html        ← frontend, served by FastAPI (required)
requirements.txt            ← Python dependencies
Dockerfile                  ← container definition
.dockerignore               ← files to exclude from build
```

---

## Railway deployment (recommended for first deploy)

1. Go to https://railway.app — sign up free with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Connect your GitHub account and select/create your repo
4. Push all the files above to your GitHub repo
5. Railway detects the Dockerfile automatically and builds it

### Set environment variables in Railway dashboard:
Go to your service → Variables → Add Variable

| Variable            | Value                            | Required |
|---------------------|----------------------------------|----------|
| `ANTHROPIC_API_KEY` | `sk-ant-...`                     | Live mode |
| `TAVILY_API_KEY`    | `tvly-...`                       | Live mode |
| `LIVE_MODE`         | `true`                           | Live mode |
| `SEARCH_PROVIDER`   | `tavily`                         | Live mode |
| `FRONTEND_ORIGIN`   | `https://your-app.up.railway.app`| Production |
| `PORT`              | (set automatically by Railway)   | Auto |

### Persistent database volume (important):
1. In Railway dashboard, go to your service → Volumes
2. Add a volume mounted at `/data`
3. The app writes `uot_runs.db` to `/data/uot_runs.db` (set via `UOT_DB_PATH` env var)
4. Without this volume, saved runs are lost on every redeploy

### Access your app:
Railway gives you a URL like `https://your-app.up.railway.app`
Open it in any browser — the frontend loads automatically.

---

## Render deployment (alternative)

1. Go to https://render.com — sign up free with GitHub
2. Click "New" → "Web Service"
3. Connect your GitHub repo
4. Set:
   - Runtime: Docker
   - Build Command: (auto-detected from Dockerfile)
   - Start Command: (auto-detected from Dockerfile CMD)
5. Set environment variables in the Render dashboard (same as Railway above)

### Persistent disk on Render:
1. Go to your service → Disks → Add Disk
2. Mount path: `/data`
3. The app will write `uot_runs.db` there

---

## Demo mode (no API keys needed)
If you just want to try the app without spending API credits:
- Do NOT set `LIVE_MODE=true`
- The app runs in demo mode with the pre-built Trump presidency scenario
- All UOT features work; only the source seeding is pre-built rather than live

---

## Quick local test before deploying
If you have Docker installed:
```bash
docker build -t uot-tee .
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e TAVILY_API_KEY="tvly-..." \
  -e LIVE_MODE=true \
  -v $(pwd)/data:/data \
  uot-tee
```
Then open http://localhost:8000 in your browser.

---

## Troubleshooting

**App deploys but shows an error on startup:**
Check logs for `ModuleNotFoundError` — make sure all five Python files are in the same directory.

**"ANTHROPIC_API_KEY not set" error:**
Add the environment variable in your Railway/Render dashboard.

**Saved runs disappear after redeploy:**
You haven't attached a persistent volume. Add one mounted at `/data`.

**CORS errors in the browser:**
Set `FRONTEND_ORIGIN` to your app's full URL (e.g. `https://your-app.up.railway.app`).

**Demo mode only, live mode not working:**
Set `LIVE_MODE=true` in environment variables AND set both API keys.
