# Repository Guidelines

## Project Structure & Module Organization
`app.py` is the Flask entry point that wires the multi-agent packages in `QueryEngine/`, `ForumEngine/`, `MediaEngine/`, `InsightEngine/`, and `ReportEngine/`. Each engine stores behavior in `agent.py`, tooling under `tools/`, and reusable nodes/prompts in dedicated subfolders. Web assets live in `templates/` and `static/`, while generated artifacts such as `final_reports/` and `logs/` stay untracked. `MindSpider/` owns the crawler CLI, `SingleEngineApp/` contains Streamlit facades for individual agents, `utils/` keeps cross-cutting helpers, and `tests/` hosts the parser regression suite.

## Build, Test, and Development Commands
```bash
# Install dependencies (pick one)
pip install -r requirements.txt
uv pip install -r requirements.txt

# Local orchestration
python app.py              # full Flask stack at http://localhost:5000
streamlit run SingleEngineApp/query_engine_streamlit_app.py --server.port 8503

# Docker workflow
docker compose up -d

# Test suite
pytest tests/test_monitor.py -v
python tests/run_tests.py   # lightweight runner for ForumEngine monitor
```

## Coding Style & Naming Conventions
Target Python 3.11, 4-space indentation, and descriptive snake_case for modules, functions, and configuration keys (`config.py` mirrors `.env`). Run `black` then `flake8` before committing; short docstrings that explain agent intent are preferred over verbose prose. New Streamlit entry points should keep the `*_engine_streamlit_app.py` suffix, and custom data/LLM adapters belong inside each engine’s `tools/` directory rather than the Flask root.

## Testing Guidelines
Pytest drives the parser coverage in `tests/test_monitor.py`; extend that module or add new files under `tests/` when touching other agents. Keep fixtures in `tests/forum_log_test_data.py` and mock network/LLM calls so CI runs remain offline. Code that affects `ForumEngine/monitor.py` must ship with fresh assertions for `process_lines_for_json` and the `extract_*` helpers; during prototyping you can run `python tests/run_tests.py` for a fast signal before executing `pytest -v`.

## Commit & Pull Request Guidelines
Create topic branches such as `feature/forum-log-rewrite` or `fix/query-cache`, follow Conventional Commit prefixes (`feat:`, `fix:`, `test:`), and keep subjects under 72 characters. PRs to `main` must summarize the change, note test evidence (`pytest -v`, `docker compose up -d` smoke), link issues, and attach UI screenshots or report snippets when front-end or Streamlit assets change. Do not commit secrets; point reviewers at `.env.example` instead of pasting credentials.

## Security & Configuration Tips
Copy `.env.example` to `.env`, fill in DB + LLM credentials, and keep the file untracked. Run `playwright install chromium` once before invoking `MindSpider/main.py --complete`, and store crawler cookies or API tokens in keychains or Docker secrets rather than code. Rotate the keys referenced in `config.py` whenever a shared environment changes hands.
