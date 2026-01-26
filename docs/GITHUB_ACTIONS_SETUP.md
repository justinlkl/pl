# GitHub Actions Automation Setup

## ✅ What's Now Automated

You now have **3 GitHub Actions workflows** running automatically:

### 1. **FPL Auto-Sync & Model Run** (`fpl-sync.yml`)
- **Runs**: Every day at 05:00 and 17:00 UTC
- **Does**:
  - Pulls latest FPL-Core-Insights data
  - Runs ensemble model projections
  - Commits & pushes results to GitHub
- **Output**: `outputs/projections.csv` (updated 2x daily)

### 2. **Weekly Update (Post-Gameweek)** (`fpl-weekly-postgw.yml`)
- **Runs**: Every 3 hours (checks if gameweek is finalized)
- **Does**:
  - Detects when current gameweek finishes
  - Syncs FPL-Core-Insights data
  - Retrains ensemble model with latest finished GW
  - Generates projections for next GW
  - Regenerates static site JSON
- **Output**: Updated ensemble model + new GW projections

### 3. **Two-Tier Training (Weekly Fine-Tune)** (`two-tier-training.yml`) ⭐ NEW
- **Runs**: Every Monday at 08:00 UTC
- **Does**:
  - Syncs FPL-Core-Insights data
  - Runs `scripts/11_weekly_finetune.py` (LSTM model fine-tuning)
  - Generates updated projections via ensemble_predict
  - Commits & pushes model + projections
- **Runtime**: ~1 minute
- **Output**: Updated `artifacts/model.keras` + `outputs/projections.csv`

---

## 🔄 Data Flow

```
GitHub (FPL-Core-Insights)
    ↓
    └─→ [GitHub Actions] Sync data
        ↓
        ├─→ [11_weekly_finetune.py] Fine-tune LSTM model
        │   ↓
        │   └─→ artifacts/model.keras
        │
        ├─→ [ensemble_predict.py] Generate projections
        │   ↓
        │   └─→ outputs/projections.csv
        │
        └─→ [Git Push] Commit results
            ↓
            └─→ GitHub repository (your repo)
                ↓
                └─→ [Streamlit App] Auto-reloads
                    (cache_data ttl=300 seconds)
```

---

## ⚙️ How to Monitor/Trigger Workflows

### View Workflow Status
1. Go to your GitHub repo: https://github.com/justinlkl/pl
2. Click **Actions** tab
3. See workflow runs + logs

### Manually Trigger Two-Tier Training
1. Go to **Actions** → **Two-Tier Training (Weekly Fine-Tune)**
2. Click **Run workflow** → **Run workflow**
3. Wait ~1 minute for completion
4. Check that `artifacts/model.keras` was updated

### Check Scheduled Runs
- **Daily sync**: Look for runs at 05:00 & 17:00 UTC
- **Post-GW updates**: Runs every 3 hours when GW is finalized
- **Weekly fine-tune**: Runs Mondays at 08:00 UTC

---

## 📊 What Gets Updated

| File | Workflow | Frequency | Trigger |
|------|----------|-----------|---------|
| `outputs/projections.csv` | fpl-sync + two-tier-training | 2x daily + weekly | Schedule |
| `artifacts/model.keras` | two-tier-training | Weekly | Monday 08:00 UTC |
| `artifacts/preprocess.joblib` | two-tier-training | Weekly | Monday 08:00 UTC |
| `artifacts/ensemble/` | fpl-weekly-postgw | Variable | Post-GW finalize |
| `site/projections.json` | fpl-sync + fpl-weekly-postgw | 2x daily + post-GW | Schedule |

---

## 🚀 Deploy Streamlit Locally

To view the dashboard with auto-updating projections:

```bash
streamlit run streamlit_app.py
```

The app will:
1. Load `outputs/projections.csv`
2. Cache it for 5 minutes
3. Auto-reload when file changes
4. Show latest player projections

---

## 🔧 If You Need to Change Timing

### Update Weekly Fine-Tune Schedule
Edit `.github/workflows/two-tier-training.yml` line 14:

```yaml
schedule:
  - cron: '0 8 * * 1'  # Monday 08:00 UTC
```

**Cron format**: `minute hour day month day-of-week`
- `0 8 * * 1` = Every Monday at 08:00 UTC
- `0 12 * * *` = Every day at 12:00 UTC
- `0 */6 * * *` = Every 6 hours

### Disable a Workflow
1. Go to **Actions**
2. Click the workflow name
3. Click **...** → **Disable workflow**

---

## ✅ Verification Checklist

- [ ] Three workflows visible in GitHub Actions
- [ ] Two-tier-training workflow runs at scheduled time
- [ ] `artifacts/model.keras` updates weekly
- [ ] `outputs/projections.csv` has current predictions
- [ ] Streamlit app loads and displays players
- [ ] Manual run of two-tier-training completes in ~1 minute

---

## 📝 Next Steps

1. **Deploy Streamlit** (if not already running):
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Monitor first automated run** (Monday 08:00 UTC):
   - Check GitHub Actions for run status
   - Verify model file timestamp updated
   - Confirm projections refreshed

3. **Optional: Deploy to cloud** (Streamlit Cloud, Heroku, etc.)

---

## 🆘 Troubleshooting

**Workflow fails to run?**
- Check GitHub Actions permissions: Settings → Actions → General → Workflow permissions (should be "Read and write")

**Model file doesn't update?**
- Check if `scripts/11_weekly_finetune.py` completes: click workflow run → view logs

**Projections look stale?**
- Manual trigger: Go to **Actions** → **Two-Tier Training** → **Run workflow**

**Git push fails?**
- Ensure workflow has write permissions
- Settings → Actions → Workflow permissions → select "Read and write permissions"
