from harness import config
from harness.determinism import check_determinism, metric_bits


def test_metric_bit_identical_twice(has_frozen_data):
    src = config.STRATEGY_FILE.read_text()
    ok, a, b = check_determinism(src)
    assert ok, f"metric differs between runs: {a!r} vs {b!r}"
    assert metric_bits(a) == metric_bits(b)


def test_synthetic_determinism(synth_series):
    src = config.STRATEGY_FILE.read_text()
    ok, a, b = check_determinism(src, synth_series)
    assert ok
