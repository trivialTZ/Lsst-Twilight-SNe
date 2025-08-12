import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.astro_utils import choose_filters_with_cap


class DummySky:
    def sky_mag(self, *args, **kwargs):
        return 21.0


def test_internal_vs_cross_changes():
    cfg = PlannerConfig(readout_s=1.0, filter_change_s=5.0,
                        exposure_by_filter={"r":10.0, "i":10.0},
                        filters=["r", "i"])
    used, timing = choose_filters_with_cap(["r", "i"], 0.0, 1000.0, cfg,
                                           current_filter="r", max_filters_per_visit=2)
    assert used == ["r", "i"]
    assert timing["cross_filter_change_s"] == 0.0
    assert timing["internal_filter_changes_s"] == cfg.filter_change_s


def test_cross_change_applied_once():
    cfg = PlannerConfig(readout_s=1.0, filter_change_s=5.0,
                        exposure_by_filter={"r":10.0}, filters=["r"])
    used, timing = choose_filters_with_cap(["r"], 0.0, 1000.0, cfg,
                                           current_filter="g")
    assert used == ["r"]
    assert timing["cross_filter_change_s"] == cfg.filter_change_s
    assert timing["internal_filter_changes_s"] == 0.0
    assert timing["total_s"] == timing["slew_s"] + timing["exposure_s"] + timing["readout_s"] + cfg.filter_change_s


def test_cap_limits_number_of_filters():
    cfg = PlannerConfig(readout_s=1.0, filter_change_s=5.0,
                        exposure_by_filter={"r":10.0, "i":10.0},
                        filters=["r", "i"])
    used, timing = choose_filters_with_cap(["r", "i"], 0.0, 15.0, cfg,
                                           max_filters_per_visit=2)
    assert used == ["r"]
    assert timing["internal_filter_changes_s"] == 0.0


def test_capped_exposure_allows_more_filters():
    cfg = PlannerConfig(readout_s=1.0, filter_change_s=0.0,
                        exposure_by_filter={"g":20.0, "r":20.0},
                        filters=["g", "r"])
    cfg.current_mag_by_filter = {"g": 10.0, "r": 10.0}
    cfg.current_alt_deg = 60.0
    cfg.sky_provider = DummySky()

    uncapped, t_uncapped = choose_filters_with_cap(["g", "r"], 0.0, 35.0, cfg,
                                                   max_filters_per_visit=2,
                                                   use_capped_exposure=False)
    capped, t_capped = choose_filters_with_cap(["g", "r"], 0.0, 35.0, cfg,
                                               max_filters_per_visit=2,
                                               use_capped_exposure=True)
    assert len(uncapped) == 1
    assert len(capped) == 2
    assert t_capped["total_s"] < t_uncapped["total_s"]
