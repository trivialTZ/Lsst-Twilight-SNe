from datetime import datetime, time, timedelta, timezone

import astropy.units as u
from astropy.coordinates import EarthLocation

from twilight_planner_pkg.astro_utils import twilight_windows_astro

LOC = EarthLocation(lat=-30.1652778 * u.deg, lon=-70.815 * u.deg, height=2215 * u.m)


def test_twilight_labeling():
    date = datetime(2024, 1, 15, tzinfo=timezone.utc)
    windows = twilight_windows_astro(date, LOC)
    assert windows[0]["label"] is None
    assert windows[-1]["label"] is None
    labels = {w["label"] for w in windows if w["label"]}
    assert labels == {"morning", "evening"}

    morning = next(w for w in windows if w["label"] == "morning")
    evening = next(w for w in windows if w["label"] == "evening")

    # Approximate local solar time; ignores civil time zones and DST
    offset_minutes = int(round((LOC.lon.to(u.deg).value / 15) * 60))
    tz = timezone(timedelta(minutes=offset_minutes))

    mid_m = morning["start"] + (morning["end"] - morning["start"]) / 2
    mid_e = evening["start"] + (evening["end"] - evening["start"]) / 2
    mid_m_local = mid_m.astimezone(tz)
    mid_e_local = mid_e.astimezone(tz)
    sunrise_local = morning["end"].astimezone(tz)
    sunset_local = evening["start"].astimezone(tz)
    local_midnight = datetime.combine(sunrise_local.date(), time(0), tz)

    assert local_midnight < mid_m_local < sunrise_local
    assert mid_e_local > sunset_local
