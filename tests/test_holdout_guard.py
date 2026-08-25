import subprocess
import sys

from harness import config


def test_runner_aborts_if_holdout_env_present(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), config.HOLDOUT_ENV_VAR: str(tmp_path)}
    r = subprocess.run([sys.executable, "runner.py", "--max-iters", "1", "--skip-report", "--proposer", "mock"],
                       cwd=config.ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert config.HOLDOUT_ENV_VAR in (r.stdout + r.stderr)


def test_no_holdout_dir_in_repo():
    assert not (config.DATA_DIR / "holdout").exists()


def test_holdout_script_refuses_inside_runner(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "CAR_RUNNER": "1", config.HOLDOUT_ENV_VAR: str(tmp_path)}
    r = subprocess.run([sys.executable, "scripts/run_holdout.py"], cwd=config.ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr)


def test_proposer_static_context_is_time_blind():
    from proposer.prompt import build_static
    text = build_static(config.PROGRAM_FILE.read_text())
    for banned in ("BTC", "ETH", "USDT", "binance", "coinbase", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"):
        assert banned not in text, banned
