# CultureSense Project Rules

## Workflow Rules

- **After any code changes**, always run `python3 build_notebook.py` to update the Kaggle notebook with the latest source code.

## Testing

- Run tests before committing:
  ```bash
  python3 test_extraction.py && python3 test_trend.py && python3 test_hypothesis.py
  ```

- Run full evaluation:
  ```bash
  python3 evaluation.py
  ```

## Code Conventions

- Keep regex patterns in `extraction.py` constants section
- All tests must pass before pushing to remote
- Document any new regex patterns added

## Common Tasks

- To debug PDF extraction: use `debug_extraction()` function from `extraction.py`
- To test locally: `python3 demo.py`
