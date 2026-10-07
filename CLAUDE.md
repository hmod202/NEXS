# NEXS

Multi-agent trading desk: a CEO agent (Claude) directs analyst agents, a code-enforced risk manager, an executor (simulator or IBKR via ib_async), and an auditor that grades every signal and rewards agents with more CEO trust.

- `nexs/desk.py`: the cycle and every agent. `nexs/broker.py`: SimBroker and IBKRBroker. `nexs/llm.py`: Claude structured-output calls (return None on failure, so agents fall back to rules). `nexs/app.py`: FastAPI, WebSocket, live-edit API. `web/index.html`: Arabic RTL dashboard, no build step.
- `config/agents.yaml` and `config/risk.yaml` are re-read every cycle, so edits apply without a restart.
- Risk limits are enforced in code (`Desk.risk_check`). Never let an LLM output bypass them.
- Tests: `python3 -m pytest -q`. Run: `./run.sh` and open http://127.0.0.1:8000.
- The UI talks Arabic to the owner; keep code and comments in English.
