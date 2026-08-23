def test_bootstrap_imports_fork():
    import _bootstrap
    assert _bootstrap.BalatroGame is not None
    g = _bootstrap.BalatroGame(seed="7I4M53DL", ruleset="mlb")
    assert g.lives == 4
