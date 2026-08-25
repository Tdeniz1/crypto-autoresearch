UV ?= uv run
STOP_AT ?= 07:00

.PHONY: help prepare determinism test smoke overnight fallback-test canary-demo dashboard report providers holdout

help:
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

prepare: ## one-time data fetch (needs CRYPTO_HOLDOUT_DIR pointing OUTSIDE the repo)
	@test -n "$$CRYPTO_HOLDOUT_DIR" || (echo "set CRYPTO_HOLDOUT_DIR=~/crypto-holdout for this command only"; exit 1)
	$(UV) python prepare.py

determinism: ## run the bit-identical metric test
	$(UV) pytest -q tests/test_determinism.py

test: ## full unit test suite
	$(UV) pytest -q

smoke: ## 10-iteration smoke test (real proposer ladder; falls back per config)
	env -u CRYPTO_HOLDOUT_DIR $(UV) python runner.py --max-iters 10 --no-llm-report

smoke-mock: ## 10-iteration offline smoke test with the mock proposer
	env -u CRYPTO_HOLDOUT_DIR $(UV) python runner.py --max-iters 10 --proposer mock --no-llm-report

overnight: ## full run until $(STOP_AT) local time (default 07:00)
	env -u CRYPTO_HOLDOUT_DIR $(UV) python runner.py --stop-at $(STOP_AT)

fallback-test: ## force rung 1 to 429 and prove the loop demotes to rung 2 and keeps going
	$(UV) python scripts/fake_429_server.py 8429 & echo $$! > /tmp/car-fake429.pid; sleep 1; \
	env -u CRYPTO_HOLDOUT_DIR OPENROUTER_API_KEY=sk-or-forced-fail CAR_OPENROUTER_BASE_URL=http://127.0.0.1:8429 CAR_BACKOFF_BASE_SEC=1 \
	  $(UV) python runner.py --max-iters 2 --rungs openrouter,claude_code --skip-report; \
	kill `cat /tmp/car-fake429.pid`; rm -f /tmp/car-fake429.pid

canary-demo: ## prove the three leakage canaries catch their bugs
	env -u CRYPTO_HOLDOUT_DIR $(UV) python scripts/canary_demo.py

dashboard: ## read-only dashboard on http://localhost:8765
	$(UV) uvicorn dashboard.app:app --host 127.0.0.1 --port 8765

report: ## regenerate the morning report for the latest session
	$(UV) python report.py --session latest

providers: ## provider attribution table across all sessions
	$(UV) python analyze_providers.py

holdout: ## manual holdout evaluation (needs CRYPTO_HOLDOUT_DIR); never run by the loop
	@test -n "$$CRYPTO_HOLDOUT_DIR" || (echo "set CRYPTO_HOLDOUT_DIR for this command only"; exit 1)
	$(UV) python scripts/run_holdout.py

# ---- live alerting (never places orders) ----
.PHONY: promote alerter-once alerter-loop alerter-status alerter-report alerter-test-push
promote: ## promote a commit: make promote SHA=971bf57 SYMBOL=BTC/USDT TF=1h
	$(UV) python -m live.promote $(SHA) --symbol $(or $(SYMBOL),BTC/USDT) --timeframe $(or $(TF),1h)
alerter-once: ## one paper tick (cron entry point)
	$(UV) python -m live.alerter run --once --mode $(or $(MODE),paper)
alerter-loop: ## paper alerter daemon: wakes after every bar close
	$(UV) python -m live.alerter run --loop --mode $(or $(MODE),paper)
alerter-status: ## state, pending acks, promoted sha
	$(UV) python -m live.alerter status
alerter-report: ## designed-vs-taken gap + skip rate → reports/alerter-<date>.md
	$(UV) python -m live.alerter_report --mode $(or $(MODE),paper)
alerter-test-push: ## verify ntfy/macOS delivery
	$(UV) python -m live.alerter test-push
