# Contributing

Install `requirements-dev.txt`, then run:

```bash
ruff check .
pytest
```

Training/model behavior changes should include tests and should update model metadata or architecture documentation when the interface changes.
