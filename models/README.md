# YOLO Model Directory

This directory should contain your trained YOLO model for fire/smoke detection.

## Required File

- `best.pt` - YOLO model weights (PyTorch format)

## How to Get the Model

### Option 1: Train Your Own Model

1. Collect and label a fire/smoke detection dataset
2. Use Ultralytics YOLO to train:
   ```bash
   yolo train data=fire_smoke.yaml model=yolov8n.pt epochs=100
   ```
3. Copy the trained `best.pt` file to this directory

### Option 2: Use Pre-trained Model

1. Download a pre-trained fire/smoke detection model
2. Ensure it's in PyTorch `.pt` format
3. Place it in this directory as `best.pt`

## Model Requirements

- **Format**: PyTorch (`.pt`)
- **Classes**: Should detect `fire` and/or `smoke`
- **Recommended**: YOLOv8 or newer
- **Input**: RGB images
- **Output**: Bounding boxes with class and confidence

## Model Configuration

The model is loaded in `backend/app.py`:

```python
yolo = FireSmokeDetector('models/best.pt')
```

If your model has a different name, update the path accordingly.

## Detection Threshold

Default confidence threshold is **0.25** (25%). Adjust in `app.py`:

```python
results = yolo.model(frame, conf=0.25, verbose=False)
```

## Model Classes

Your model should have classes similar to:
```python
{
    0: 'fire',
    1: 'smoke'
}
```

The system will log the model classes on startup.

## Important Notes

⚠️ **Model files are NOT included in the Git repository** due to size constraints.

- Model files (`.pt`) are excluded via `.gitignore`
- Users must add their own trained model
- Consider using Git LFS for large files if needed
- Or host model files separately (Google Drive, S3, etc.)

## Testing Your Model

After adding the model:

1. Start the backend server:
   ```bash
   uvicorn app:app --reload
   ```

2. Check console output for model info:
   ```
   🔥 Fire/Smoke Detection Model Loaded: models/best.pt
   📋 Model classes: {0: 'fire', 1: 'smoke'}
   ```

3. Test with a fire image/video in the frontend

## Troubleshooting

### Model not found
```
FileNotFoundError: models/best.pt
```
**Solution**: Add your model file to this directory

### Wrong model format
```
Error loading model
```
**Solution**: Ensure model is in PyTorch `.pt` format, not ONNX or TensorFlow

### Poor detection accuracy
- Try lowering confidence threshold
- Retrain model with more diverse dataset
- Use higher resolution images
- Ensure good lighting in test images

---

**Need help training a model?** Check the [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
