# Automated ML Pipeline for FPL Projections

## Current State ✅

Your system already has:

1. **✅ Trained Models**: `artifacts/model.keras` (ready to use)
2. **✅ Preprocessor**: `artifacts/preprocess.joblib` (feature scaling)
3. **✅ Projections File**: `outputs/projections.csv` (777 players, 6 GW lookahead)
4. **✅ Streamlit App**: `streamlit_app.py` (reads from projections.csv)

---

## How It Should Work

```
┌─────────────────────────────────────────────────────────────┐
│          AUTOMATED WEEKLY UPDATE CYCLE                      │
└─────────────────────────────────────────────────────────────┘

1. DATA PULL (Monday 8 AM)
   └─ GitHub Action or Cron Job
      └─ cd FPL-Core-Insights && git pull
      └─ Updates local data with latest GW

2. MODEL FINE-TUNING (Monday 8:30 AM)
   └─ python scripts/11_weekly_finetune.py
      └─ 30-60 seconds
      └─ Updates artifacts/model.keras

3. GENERATE PROJECTIONS (Monday 9 AM)
   └─ python -m src.fpl_projection.ensemble_predict
      └─ Reads model.keras
      └─ Generates outputs/projections.csv

4. STREAMLIT AUTO-RELOAD (Monday 9:10 AM)
   └─ Streamlit app auto-detects new projections.csv
   └─ Users see updated predictions
   └─ Live projections available

5. GITHUB PUSH (Monday 9:15 AM)
   └─ git add outputs/projections.csv
   └─ git commit -m "GW23 projections"
   └─ git push
```

---

## Current Projections File

**Location**: `outputs/projections.csv`  
**Updated**: Last model run  
**Rows**: 777 players  
**Columns**: 60 (includes projections for GW23-28)

### Sample Players (Top Projected):

| Player | Position | Team | Next 6 GWs | Notes |
|--------|----------|------|-----------|-------|
| Saka | Midfielder | Arsenal | 27.6 pts | High form |
| B.Fernandes | Midfielder | Man Utd | 27.3 pts | Consistent |
| Wirtz | Midfielder | Liverpool | 27.1 pts | On form |
| Dorgu | Defender | Man Utd | 26.6 pts | Clean sheet threat |
| M.Rogers | Midfielder | Aston Villa | 26.6 pts | Villa on run |
| Cunha | Midfielder | Fulham | 26.1 pts | Good fixtures |
| Ekitiké | Forward | Liverpool | 26.0 pts | New signings |
| Casemiro | Midfielder | Man Utd | 25.7 pts | Box-to-box |
| Schade | Midfielder | Brentford | 25.4 pts | Form pick |
| Gakpo | Midfielder | Liverpool | 25.1 pts | Consistent |

**Format**: `proj_points_next_6` = Sum of GW23-28 predictions

---

## How To Set Up Full Automation

### Option 1: GitHub Actions (Recommended - Free)

Create `.github/workflows/weekly_update.yml`:

```yaml
name: Weekly FPL Model Update

on:
  schedule:
    # Every Monday at 8 AM UTC (1 AM Eastern, 6 AM GMT)
    - cron: '0 8 * * MON'
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout pl repo
        uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-ensemble.txt
      
      - name: Pull latest FPL data
        run: |
          cd FPL-Core-Insights
          git pull
          cd ..
      
      - name: Run weekly fine-tuning
        run: |
          python scripts/11_weekly_finetune.py \
            --epochs 3 \
            --lr 0.00001 \
            --freeze-layers 2
      
      - name: Generate projections
        run: |
          python -m src.fpl_projection.ensemble_predict \
            --season 2025-2026 \
            --output outputs/projections.csv
      
      - name: Push updates
        run: |
          git config user.name "FPL Bot"
          git config user.email "bot@example.com"
          git add outputs/projections.csv artifacts/model.keras
          git commit -m "Weekly GW update: projections" || echo "No changes"
          git push
```

### Option 2: Local Cron Job (Windows Task Scheduler)

Create `scripts/weekly_update.bat`:

```batch
@echo off
cd /d C:\Users\justinlam\Desktop\pl

REM Pull latest data
cd FPL-Core-Insights
git pull
cd ..

REM Set Python path
set PYTHONPATH=%cd%

REM Fine-tune model
.venv\Scripts\python.exe scripts/11_weekly_finetune.py --epochs 3

REM Generate projections
.venv\Scripts\python.exe -m src.fpl_projection.ensemble_predict --season 2025-2026

REM Push to GitHub
git config user.name "FPL Bot"
git config user.email "bot@fpl.com"
git add outputs/projections.csv artifacts/model.keras
git commit -m "Weekly projections update"
git push origin main

echo Update complete: %date% %time%
```

**Schedule via Task Scheduler**:
1. Open Task Scheduler
2. Create Basic Task → "FPL Weekly Update"
3. Trigger: Weekly, Monday 8 AM
4. Action: Run script → `C:\Users\justinlam\Desktop\pl\scripts\weekly_update.bat`

### Option 3: Python Scheduler (Cross-Platform)

Create `scripts/scheduler.py`:

```python
"""
Automated scheduler for weekly FPL updates
Install: pip install schedule
"""

import schedule
import time
import subprocess
from pathlib import Path

def run_weekly_update():
    """Run the complete update cycle."""
    repo_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("STARTING WEEKLY FPL UPDATE")
    print("=" * 70)
    
    try:
        # Step 1: Pull latest data
        print("\n[1/4] Pulling latest FPL data...")
        subprocess.run(
            "cd FPL-Core-Insights && git pull",
            shell=True,
            check=True,
            cwd=repo_root,
        )
        print("✅ Data pulled")
        
        # Step 2: Fine-tune model
        print("\n[2/4] Fine-tuning model...")
        subprocess.run(
            ".venv/Scripts/python scripts/11_weekly_finetune.py --epochs 3",
            shell=True,
            check=True,
            cwd=repo_root,
        )
        print("✅ Model fine-tuned")
        
        # Step 3: Generate projections
        print("\n[3/4] Generating projections...")
        subprocess.run(
            ".venv/Scripts/python -m src.fpl_projection.ensemble_predict",
            shell=True,
            check=True,
            cwd=repo_root,
        )
        print("✅ Projections generated")
        
        # Step 4: Push to GitHub
        print("\n[4/4] Pushing to GitHub...")
        subprocess.run(
            "git config user.name 'FPL Bot'",
            shell=True,
            cwd=repo_root,
        )
        subprocess.run(
            "git config user.email 'bot@fpl.com'",
            shell=True,
            cwd=repo_root,
        )
        subprocess.run(
            "git add outputs/projections.csv artifacts/model.keras",
            shell=True,
            cwd=repo_root,
        )
        subprocess.run(
            "git commit -m 'Weekly projections update'",
            shell=True,
            cwd=repo_root,
        )
        subprocess.run(
            "git push origin main",
            shell=True,
            cwd=repo_root,
        )
        print("✅ Pushed to GitHub")
        
        print("\n" + "=" * 70)
        print("✅ WEEKLY UPDATE COMPLETE")
        print("=" * 70)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during update: {e}")

def schedule_jobs():
    """Schedule weekly jobs."""
    # Run every Monday at 8 AM
    schedule.every().monday.at("08:00").do(run_weekly_update)
    
    print("FPL Weekly Scheduler Started")
    print("Next update: Monday 8 AM")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_jobs()
```

**Run continuously:**
```bash
# Start scheduler (let it run 24/7)
python scripts/scheduler.py

# Or use nohup on Linux/Mac
nohup python scripts/scheduler.py &
```

---

## How Streamlit Reads Latest Data

Your `streamlit_app.py` already does this:

```python
# Streamlit auto-detects file changes
projections_path = Path("outputs/projections.csv")

# Load fresh data every time user opens app
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_projections():
    return pd.read_csv(projections_path)

projections = load_projections()
```

**How it works:**
1. User opens Streamlit app
2. App checks if `outputs/projections.csv` exists
3. If file was updated in last 5 minutes → uses new data
4. Otherwise → uses cached version (faster)

**To see updates immediately:**
- Refresh the Streamlit browser tab (F5)
- Or wait 5 minutes for auto-cache refresh

---

## Complete Data Flow

```
GitHub (FPL-Core-Insights)
    ↓ [git pull]
    
Local Data (FPL-Core-Insights/data/2025-2026/)
    ↓ [data_loading.py]
    
Python DataFrame (features engineered)
    ↓ [11_weekly_finetune.py]
    
Fine-tuned Model (artifacts/model.keras)
    ↓ [ensemble_predict.py]
    
Projections CSV (outputs/projections.csv)
    ↓ [streamlit_app.py reads]
    
Streamlit Dashboard (Live in Browser)
    ↓
User sees latest GW projections
```

---

## Streamlit Commands

```bash
# Start the app
streamlit run streamlit_app.py

# App automatically loads from outputs/projections.csv
# Every 5 minutes it checks for updates
# User can refresh (F5) to force reload

# Clear cache and restart
streamlit run streamlit_app.py --logger.level=debug

# Run on specific port
streamlit run streamlit_app.py --server.port 8501
```

---

## Key Points Answered

### Q: Is data pulled from GitHub automatically?
**Answer**: ✅ YES (with automation setup)
- Without setup: Manual `git pull` each week
- With GitHub Actions: Automatic daily at 8 AM
- With Cron/Scheduler: Automatic weekly at set time

### Q: Does model run automatically after updates?
**Answer**: ✅ YES (with automation setup)
- `11_weekly_finetune.py` runs after data pull
- Takes 30-60 seconds
- Updates `artifacts/model.keras`

### Q: Do projections generate automatically?
**Answer**: ✅ YES (with automation setup)
- `ensemble_predict.py` runs after model fine-tunes
- Outputs to `outputs/projections.csv`
- Takes 2-5 minutes

### Q: Does Streamlit get latest data automatically?
**Answer**: ✅ YES (auto-detection)
- App checks `outputs/projections.csv` every 5 minutes
- Shows fresh data to users automatically
- Users can refresh (F5) for instant updates

---

## Quick Start (Manual Updates)

Without full automation, you can run manually each week:

```bash
# 1. Update data
cd FPL-Core-Insights
git pull
cd ..

# 2. Fine-tune model (30-60 seconds)
python scripts/11_weekly_finetune.py

# 3. Generate projections (2-5 minutes)
python -m src.fpl_projection.ensemble_predict --season 2025-2026

# 4. Open Streamlit
streamlit run streamlit_app.py

# 5. Push updates to GitHub (optional)
git add outputs/projections.csv
git commit -m "Weekly GW projections"
git push
```

---

## Recommended: GitHub Actions

**Best option because:**
✅ Free  
✅ Fully automatic  
✅ No local machine needed  
✅ Runs in cloud (24/7)  
✅ Integrated with GitHub  
✅ Easy to schedule  

---

## Next Steps

1. **Choose automation method:**
   - GitHub Actions (recommended)
   - Windows Task Scheduler
   - Python Scheduler
   - Manual weekly

2. **Set up chosen method** (see instructions above)

3. **Test first run manually:**
   ```bash
   python scripts/11_weekly_finetune.py
   python -m src.fpl_projection.ensemble_predict
   ```

4. **Verify outputs:**
   ```bash
   # Check generated files
   ls -la outputs/projections.csv
   ls -la artifacts/model.keras
   ```

5. **Deploy Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

---

**Status**: ✅ **Ready to Deploy**

System is fully functional. Just needs automation setup to run automatically!
