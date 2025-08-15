# hydraulics-prediction-software
Predictive optimization for industrial hydraulic systems — paper + code for efficiency gain modeling and setpoint recommendations
# Hydraulic Efficiency Optimizer

Predictive optimization framework for industrial hydraulic systems.  
**Paper:** `paper/hydraulic_efficiency_preprint.pdf` (also on Zenodo/OSF).

## Features
- Efficiency-gain prediction from operating parameters (pressure, flow, temp, RPM, load)
- Simple recommendations for improved setpoints
- Reproducible experiments via notebooks
- 
### Contents
- **paper/** – Preprint PDF of the research study.
- **src/** – Core Python source code for the efficiency predictor and optimizer.
- **app/** – Application interface files for running the model as a standalone tool.
- **model/** – Trained model files and related scripts.
- **data/** – Sample datasets and data usage instructions.
- **notebooks/** – Jupyter/Colab notebooks for experimentation and reproducibility.
- 
## Quick Start
```bash
pip install -r requirements.txt
