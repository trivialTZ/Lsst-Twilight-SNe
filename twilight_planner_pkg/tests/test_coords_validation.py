import pathlib, sys
import pytest

# ensure package root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from twilight_planner_pkg.astro_utils import validate_coords


def test_validate_coords_ok():
    ra, dec = validate_coords(10, 89.9999999)
    assert ra == pytest.approx(10)
    assert dec == pytest.approx(89.9999999)


def test_validate_coords_wrap():
    ra, dec = validate_coords(-1, 0)
    assert 0 <= ra < 360
    assert dec == 0


def test_validate_coords_error():
    with pytest.raises(ValueError):
        validate_coords(10, 100)
