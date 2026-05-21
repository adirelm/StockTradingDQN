"""Tests for the config loader (B1/C5 — no hardcoded values; deck param names)."""

import pytest

from tradedqn.config import Config, assert_in_project, load_config

CONFIG_PATH = "config/config.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG_PATH)


class TestProjectConfig:
    def test_state_window_matches_deck(self, cfg):
        assert cfg.features.window_size == 30
        assert cfg.features.features_count == 10

    def test_ten_feature_names(self, cfg):
        assert len(cfg.features.names) == 10
        assert cfg.features.names[0] == "log_return"
        assert cfg.features.names[1] == "rsi_14"
        assert cfg.features.names[8:] == ["position", "unrealized_pnl"]

    def test_action_encoding_matches_deck(self, cfg):
        assert (cfg.actions.sell, cfg.actions.hold, cfg.actions.buy) == (0, 1, 2)

    def test_training_hyperparameters(self, cfg):
        assert cfg.training.gamma == 0.95
        assert cfg.training.epsilon_start > cfg.training.epsilon_min
        assert cfg.network.conv_channels == [32, 64]


class TestLoaderBehaviour:
    def test_nested_attribute_access(self):
        cfg = Config({"a": {"b": {"c": 7}}})
        assert cfg.a.b.c == 7

    def test_to_dict_round_trips(self):
        data = {"x": 1, "nested": {"y": [1, {"z": 2}]}}
        assert Config(data).to_dict() == data

    def test_relative_path_resolves_from_project_root(self, monkeypatch, tmp_path):
        # loader must not depend on the current working directory
        monkeypatch.chdir(tmp_path)
        assert load_config(CONFIG_PATH).seed == 42

    def test_non_mapping_root_raises(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_config(str(bad))

    def test_project_config_has_version(self, cfg):
        assert str(cfg.version).startswith("1.")

    def test_missing_version_key_raises(self, tmp_path):
        f = tmp_path / "nover.yaml"
        f.write_text("seed: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing the required 'version'"):
            load_config(str(f))

    def test_incompatible_major_version_raises(self, tmp_path):
        f = tmp_path / "old.yaml"
        f.write_text('version: "2.0.0"\nseed: 1\n', encoding="utf-8")
        with pytest.raises(ValueError, match="incompatible"):
            load_config(str(f))

    def test_missing_config_file_has_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run TradeDQN from the project"):
            load_config(str(tmp_path / "nope.yaml"))


class TestAssertInProject:
    def test_absolute_path_passes_through(self, tmp_path):
        p = str(tmp_path / "x.pt")
        assert assert_in_project(p) == p

    def test_relative_inside_project_resolves(self):
        assert assert_in_project("results/x.pt").endswith("results/x.pt")

    def test_relative_escape_refused(self):
        with pytest.raises(ValueError, match="outside the project root"):
            assert_in_project("../../../etc/evil")
