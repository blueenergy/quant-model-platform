from quant_model_platform.evaluation.walk_forward import purged_walk_forward_folds


def test_purged_walk_forward_has_embargo():
    dates = [f"2024-01-{d:02d}" for d in range(1, 31)]
    folds = purged_walk_forward_folds(dates, train_days=10, test_days=5, purge_days=3)
    assert folds
    f0 = folds[0]
    assert len(f0.purge_dates) == 3
    assert f0.train_dates[-1] < f0.purge_dates[0]
    assert f0.purge_dates[-1] < f0.test_dates[0]
