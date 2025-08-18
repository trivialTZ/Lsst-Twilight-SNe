import pathlib
import sys

# Ensure package root is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pytest

from twilight_planner_pkg.main import build_parser
from twilight_planner_pkg.main import main as entry_main
from twilight_planner_pkg.main import parse_exp_map, parse_filters


def test_build_parser_required_args_present():
    parser = build_parser()
    required = {"csv", "out", "start", "end", "lat", "lon", "height"}
    actions = {a.dest: a for a in parser._actions}
    for name in required:
        assert name in actions, f"Missing {name} argument"
        assert actions[name].required is True


def test_parse_exp_map_valid_and_empty():
    assert parse_exp_map("g:5,r:10") == {"g": 5.0, "r": 10.0}
    assert parse_exp_map("") == {}


def test_parse_exp_map_malformed():
    with pytest.raises(ValueError):
        parse_exp_map("g:5,r10")
    with pytest.raises(ValueError):
        parse_exp_map("g:five")


def test_parse_filters_string_and_csv():
    assert parse_filters("g,r") == ["g", "r"]
    assert parse_filters("gr") == ["g", "r"]


def test_main_smoke(tmp_path, monkeypatch):
    csv_file = tmp_path / "input.csv"
    csv_file.write_text("id\n1\n")
    outdir = tmp_path / "out"

    called = {}

    def fake_plan(
        csv_path, outdir, start_date, end_date, cfg, run_label="hybrid", verbose=True
    ):
        called["csv_path"] = csv_path
        called["outdir"] = outdir
        called["cfg"] = cfg
        called["run_label"] = run_label
        return None

    monkeypatch.setattr(
        "twilight_planner_pkg.main.plan_twilight_range_with_caps", fake_plan
    )
    args = [
        "--csv",
        str(csv_file),
        "--out",
        str(outdir),
        "--start",
        "2024-01-01",
        "--end",
        "2024-01-02",
        "--lat",
        "0",
        "--lon",
        "0",
        "--height",
        "0",
        "--filters",
        "xxx",
        "--evening-twilight",
        "18:00",
        "--morning-twilight",
        "05:00",
    ]
    monkeypatch.setattr(sys, "argv", ["prog"] + args)
    entry_main()
    assert called["csv_path"] == str(csv_file)
    assert called["outdir"] == str(outdir)
    assert called["cfg"].filters == ["x", "x", "x"]
    assert called["cfg"].evening_twilight == "18:00"
    assert called["cfg"].morning_twilight == "05:00"
    assert called["run_label"] == "hybrid"
    assert outdir.exists()
