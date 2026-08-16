# Adaptive Payment Security Range: empirical falsification spike

This isolated experiment tests five preregistered claims about the proposed architecture. It deliberately uses a small transparent simulator and NumPy logistic regression so that the result is reproducible with the libraries already available in the workspace.

## Reproduce

From this directory:

```bash
/Users/dylanmoraes/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 src/validate.py
```

The script runs five fixed seeds and writes:

- `outputs/results.json`: machine-readable aggregate and per-seed results
- `outputs/metrics.csv`: flat metric table
- `outputs/console.log`: exact run summary
- `outputs/run_history.json`: code hashes and execution status
- `empirical_report.md`: concise interpretation against H1-H5

Expected runtime is under two minutes on a laptop. No network access, external service, or private data is used.

## Environment

- Python 3.12.13 (Codex bundled workspace runtime)
- NumPy 2.3.5
- pandas 2.2.3

The implementation does not require scikit-learn, LightGBM, CatBoost, SciPy, or a GPU.

## Boundaries

This is a synthetic architecture test, not validation against Mastercard or production payment data. A positive result only establishes that the architectural mechanisms can be implemented and that they work under declared simulated shifts. It cannot establish real-world transfer, calibration, or commercial readiness.
