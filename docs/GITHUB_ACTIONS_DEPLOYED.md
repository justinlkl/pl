# ✅ GitHub Actions Automation Complete

## What Was Deployed

A new GitHub Actions workflow **`two-tier-training.yml`** is now active in your repository.

### Workflow Details
- **Name**: Two-Tier Training (Weekly Fine-Tune)
- **Schedule**: Every Monday at 08:00 UTC
- **Location**: [.github/workflows/two-tier-training.yml](.github/workflows/two-tier-training.yml)

### Automation Steps
1. ✅ Sync FPL-Core-Insights data
2. ✅ Run `scripts/11_weekly_finetune.py` (LSTM model fine-tuning)
3. ✅ Generate projections via `ensemble_predict`
4. ✅ Commit & push updated model + projections
5. ✅ Streamlit auto-reloads with new data

---

## Your Complete Automation Stack

You now have **3 workflows** providing full automation:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| **fpl-sync.yml** | Every 3 hours (& 05:00/17:00 UTC) | Data sync + ensemble predictions |
| **fpl-weekly-postgw.yml** | Every 3 hours (on post-GW) | Retrain ensemble when GW finishes |
| **two-tier-training.yml** ⭐ NEW | Monday 08:00 UTC | LSTM fine-tuning + fresh projections |

---

## 🚀 Next Steps

### 1. Monitor First Run (Monday 08:00 UTC)
Visit: https://github.com/justinlkl/pl/actions
- Click **Two-Tier Training (Weekly Fine-Tune)**
- Watch the run execute
- Verify success (green checkmark)

### 2. Deploy Streamlit App (Local or Cloud)

**Local:**
```bash
streamlit run streamlit_app.py
```

**Cloud (Recommended):**
1. Push repo to GitHub ✅ (done)
2. Go to https://streamlit.io/cloud
3. Click "Create App" → point to your repo
4. Select `streamlit_app.py`
5. Deploy in ~2 minutes

### 3. Optional: Customize Schedule

Edit `.github/workflows/two-tier-training.yml` line 14 if you want different timing:

```yaml
schedule:
  - cron: '0 8 * * 1'  # Monday 08:00 UTC
  # Change to e.g.: '0 12 * * *' for daily 12:00 UTC
```

---

## 📊 What Gets Auto-Updated

- **Every Monday at 08:00 UTC**:
  - `artifacts/model.keras` ← LSTM model fine-tuned
  - `artifacts/preprocess.joblib` ← Updated scaler
  - `outputs/projections.csv` ← 777 players' next 6-GW projections
  - `outputs/projections_internal.csv` ← Internal debugging file

- **Streamlit app** auto-detects file changes + 5-min cache TTL

---

## 🎯 You Can Now:

✅ **Stop worrying about manual updates** - workflows handle it  
✅ **Get fresh player projections weekly** - Monday morning  
✅ **Display latest data in Streamlit** - auto-reload on file change  
✅ **Monitor in GitHub Actions** - see logs, trigger manually  

---

## 📝 Documentation

Full setup details: [docs/GITHUB_ACTIONS_SETUP.md](../docs/GITHUB_ACTIONS_SETUP.md)

---

## 🆘 If Something Breaks

1. **Check GitHub Actions logs**: https://github.com/justinlkl/pl/actions
2. **Common issues**:
   - Script fails → Check Python dependencies (pip install -r requirements.txt)
   - Push fails → Check workflow permissions (Settings → Actions)
   - No file updates → Check if workflow ran (Actions tab)

3. **Quick fix**: Manually trigger workflow
   - Go to **Actions** → **Two-Tier Training** → **Run workflow**

---

**Status**: ✅ LIVE  
**Last updated**: $(date -u)  
**Automation ready for production**
