# mlops-pipeline

Minimal MLOps pipeline scaffold.

Commands:

Create virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn api.main:app --reload
```
