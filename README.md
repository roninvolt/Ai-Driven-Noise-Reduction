<<<<<<< HEAD
# ai-driven-noise-reduction
AI-based automatic noise classification and adaptive denoising system with SAR image extension 
=======
# Ai-Driven-Noise-Reduction

AI-based automatic noise classification and adaptive denoising system with SAR image extension 
>>>>>>> 9ee92f778e760738ffaf817fa9631f8e2432e307

## Project structure

```text
src/
  noise_classifier/   # classifier, dataset, generator, transforms
  denoiser/           # AI-first denoiser modules + router + inference helpers
  pipeline/           # run_pipeline package
  metrics/            # psnr.py and ssim.py
  ui/                 # Streamlit application
legacy/
  classical_denoise/  # legacy baseline denoisers
scripts/
  train_noise_classifier.py
  eval_noise_classifier.py
  run_pipeline.py
  run_ui.py
```

## Project Architecture

The active path is AI-first:
1. `noise_classifier` predicts the noise type with model-first + heuristic fallback.
2. `pipeline.run_pipeline` orchestrates classification and denoiser routing.
3. `denoiser.router` selects a noise-specific denoiser backend.

`legacy/classical_denoise` remains as baseline compatibility while AI denoiser models are being trained.

## Quick start

```bash
python -m pip install -r requirements.txt
python test.py
python scripts/run_ui.py
```

## Training Noise Classifier

Local:

```bash
python scripts/train_noise_classifier.py --data_dir dataset/synthetic --out_dir models --epochs 10 --batch_size 32 --device cpu
python scripts/eval_noise_classifier.py --checkpoint models/noise_classifier_best.pt --data_dir dataset/synthetic --device cpu
```

Colab:

```bash
!python scripts/train_noise_classifier.py --data_dir /content/dataset/synthetic --out_dir /content/models --epochs 10 --batch_size 32 --device cuda
!python scripts/eval_noise_classifier.py --checkpoint /content/models/noise_classifier_best.pt --data_dir /content/dataset/synthetic --device cuda
```
