# Deployment Instructions for Render.com

## Issues Fixed ✓

1. **Model Path Issue**: Fixed path mismatch between where model is saved (`flask/payments.pkl`) and where Flask loads it
2. **Gunicorn Missing**: Pinned gunicorn version to 23.0.0 in requirements.txt
3. **Procfile Missing**: Created Procfile with correct start command
4. **Dependency Versions**: Pinned all package versions to prevent compatibility issues

## Before Deploying

### IMPORTANT: Train the Model First! 

**You MUST train and generate the model file before deployment:**

```bash
# Run one of these commands in the terminal:
python train_model_quick.py    # Fastest option (recommended for testing)
# OR
python train_model_fast.py     # Medium speed
# OR  
python train_model.py          # Full training (slower but more accurate)
```

This will create the `flask/payments.pkl` file that the web app needs.

### Verify the Model Exists

After training, confirm the model file exists:
```bash
dir flask\payments.pkl
```

## Deploy to Render

1. **Commit all changes**:
   ```bash
   git add .
   git commit -m "Fix deployment: add Procfile, pin gunicorn, fix model path"
   git push
   ```

2. **In Render Dashboard**:
   - The build should automatically detect the Procfile
   - Build Command: `pip install -r requirements.txt`
   - Start Command: Should auto-detect from Procfile: `gunicorn flask.app:app`

3. **Important**: Make sure `flask/payments.pkl` is committed to your repository (if file size < 100MB)
   - Check .gitignore doesn't exclude .pkl files
   - If model is too large, consider using Git LFS or training on Render during build

## What Was Changed

### [flask/app.py](flask/app.py)
- Added `import os`
- Changed model loading to use proper path resolution

### [flask/app_ibm.py](flask/app_ibm.py)  
- Added `import os`
- Changed model loading to use proper path resolution

### [requirements.txt](requirements.txt)
- Removed jupyter/notebook (not needed for production)
- Pinned all package versions for reproducibility
- Ensured gunicorn==23.0.0 is included

### [Procfile](Procfile) (NEW)
- Tells Render how to start the application
- Command: `web: gunicorn flask.app:app`

## Troubleshooting

### If deployment still fails:

1. **Check model file exists in repository**:
   ```bash
   git ls-files | findstr payments.pkl
   ```

2. **If model file is too large for git**:
   - Add training as a build step in Render
   - Or use Git LFS
   - Or use external storage (S3, etc.)

3. **Check Render logs** for specific errors

4. **Test locally first**:
   ```bash
   python train_model_quick.py
   gunicorn flask.app:app
   # Then visit http://localhost:8000
   ```

## Next Steps

1. Train the model with one of the training scripts
2. Commit and push all changes
3. Redeploy on Render
4. Monitor the deployment logs

Your deployment should now succeed! 🚀
