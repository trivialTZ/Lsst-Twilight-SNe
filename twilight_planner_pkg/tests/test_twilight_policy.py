import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from twilight_planner_pkg.astro_utils import allowed_filters_for_sun_alt
from twilight_planner_pkg.config import PlannerConfig


def test_twilight_policy_sets():
    cfg = PlannerConfig()
    for alt, expected in [
        (-16, ["y", "z", "i"]),
        (-13, ["z", "i", "r"]),
        (-5, ["i", "z", "y"]),
    ]:
        allowed = allowed_filters_for_sun_alt(alt, cfg)
        assert allowed == expected
        assert set(allowed).issubset(set(cfg.filters))


def test_custom_policy_monotonic_reddening():
    policy = [(-18.0, -15.0, ["g", "r", "i"]), (-15.0, 0.0, ["i", "z", "y"])]
    cfg = PlannerConfig(sun_alt_policy=policy)
    dark = allowed_filters_for_sun_alt(-16.0, cfg)
    bright = allowed_filters_for_sun_alt(-5.0, cfg)
    order = {f: i for i, f in enumerate(["u", "g", "r", "i", "z", "y"])}
    assert min(order[f] for f in bright) >= min(order[f] for f in dark)
    assert set(dark).issubset(set(cfg.filters))
    assert set(bright).issubset(set(cfg.filters))
