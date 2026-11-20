import json

import pytest

from molgnn.configuration import Configuration
from molgnn.data import PERIODIC_TABLE_SYMS, PERIODIC_TABLE_Z
from molgnn.utils import str_to_float


@pytest.mark.parametrize(
    "text, value",
    [("1.5", 1.5), ("-0.3877", -0.3877), ("0.", 0.0), ("1.5*^-5", 1.5e-5), ("-2.1*^3", -2100.0)],
)
def test_str_to_float(text, value):
    assert str_to_float(text) == pytest.approx(value)


def test_periodic_tables_are_consistent():
    for z, entry in PERIODIC_TABLE_Z.items():
        assert PERIODIC_TABLE_SYMS[entry["symbol"]] == z


def test_get_missing_file_returns_default():
    assert Configuration.get("nope", 42) == 42


def test_set_save_load_roundtrip(tmp_path):
    Configuration.set("lr", 0.01)
    Configuration.set("layers", [1, 2], auto_save=True)
    on_disk = json.loads((tmp_path / "configuration.json").read_text())
    assert on_disk == {"lr": 0.01, "layers": [1, 2]}

    Configuration._config = {}
    Configuration.load()
    assert Configuration.get("lr") == 0.01


def test_set_filepath_loads_new_file(tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"channels": 7}))
    Configuration.set_filepath(other)
    assert Configuration.get("channels") == 7


def test_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        Configuration.set_filepath(bad)
