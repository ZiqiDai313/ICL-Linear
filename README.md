# Training transformer model for ICL linear system solving

## 1. Install PyHessian package from local source

```bash
conda activate your_env
cd ./PyHessian
pip install -e .
```

## 2. Generate evaluation data

```bash
python test_solver_tune_data.py
```

## 3. Run training script

```bash
bash slurm_run_training.sh 1
```

