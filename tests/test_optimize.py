from vcm_preprocess import BDRateSearch, PreprocessConfig

def test_bd_search_runs():
    ref = [(100, .5), (200, .6), (400, .7), (800, .8)]
    def evaluator(cfg: PreprocessConfig):
        gain = 0.9 if cfg.sharpen > .5 else 1.0
        return [(r * gain, q) for r, q in ref]
    result = BDRateSearch(ref, evaluator, iterations=5, seed=1).run()
    assert result.metrics["bd_rate_percent"] <= 0
