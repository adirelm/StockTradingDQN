"""TradingSDK — the single facade the UIs call (§4 SDK mandate).

Wires Data → Preprocessor → split/normalize → Env → Agent →
Training/Backtest/Inference behind one clean API. The terminal and GUI
interfaces depend on this object only; they never touch the engine modules.
"""

from __future__ import annotations

from tradedqn.config import DEFAULT_CONFIG_PATH, Config, assert_in_project, load_config
from tradedqn.data.client import DataClient
from tradedqn.data.gatekeeper import RateLimitGatekeeper
from tradedqn.env.trading_env import TradingEnvironment, assemble_state
from tradedqn.features.dataset import MinMaxNormalizer, chronological_split
from tradedqn.features.preprocessor import Preprocessor
from tradedqn.model.agent import DQNAgent
from tradedqn.services.backtest import BacktestService
from tradedqn.services.inference import InferenceService
from tradedqn.services.training import TrainingService


class TradingSDK:
    def __init__(self, config_path=None, cfg=None, data_client=None, agent=None) -> None:
        self.cfg = cfg or load_config(config_path or DEFAULT_CONFIG_PATH)
        self.data_client = data_client or DataClient(self.cfg.data.cache_dir, self._gatekeeper())
        self.preprocessor = Preprocessor(self.cfg.features)
        self.agent = agent or DQNAgent(self.cfg)
        self.inference = InferenceService.from_config(self.agent, self.cfg)
        self._splits: dict | None = None

    def _gatekeeper(self) -> RateLimitGatekeeper:
        r = self.cfg.data.rate_limit
        return RateLimitGatekeeper(
            r.min_interval_seconds, r.max_calls_per_window, r.window_seconds, r.max_retries
        )

    def prepare_data(self, ticker=None, start=None, end=None) -> dict[str, int]:
        """Fetch → features → chronological split → normalize (fit on train).

        ``ticker``/``start``/``end`` override the config defaults so a UI can let
        the user choose the symbol and date range; ``None`` keeps the config value.
        """
        d = self.cfg.data
        ohlcv = self.data_client.get_ohlcv(
            ticker or d.ticker, start or d.start, end or d.end, d.interval
        )
        features = self.preprocessor.compute(ohlcv)
        prices = ohlcv.loc[features.index, "Close"]
        s = self.cfg.split
        train_f, val_f, test_f = chronological_split(features, s.train, s.validation, s.test)
        normalizer = MinMaxNormalizer().fit(train_f)
        self._splits = {
            name: (normalizer.transform(frame), prices.loc[frame.index])
            for name, frame in (("train", train_f), ("validation", val_f), ("test", test_f))
        }
        return {name: len(frame) for name, (frame, _) in self._splits.items()}

    def _require_data(self) -> None:
        if self._splits is None:
            raise RuntimeError("call prepare_data() before train/backtest/recommend")

    def _env(self, split: str) -> TradingEnvironment:
        features, prices = self._splits[split]
        return TradingEnvironment(features, prices.to_numpy(), self.cfg)

    def train(self, episodes: int | None = None, on_episode=None) -> list[dict]:
        self._require_data()
        episodes = episodes if episodes is not None else self.cfg.training.episodes
        return TrainingService(self._env("train"), self.agent).train(episodes, on_episode=on_episode)

    def backtest(self, split: str = "test") -> dict:
        self._require_data()
        return BacktestService(self._env(split), self.agent).run()

    def _cfg_with_dueling(self, dueling: bool) -> Config:
        data = self.cfg.to_dict()
        data["network"]["dueling"] = dueling
        return Config(data)

    def compare(self, episodes: int | None = None, on_episode=None) -> dict[str, list]:
        """Train a Dueling and a plain DQN on the same data → per-arch histories.

        The §9 ablation: same trunk, only the dueling head differs. ``on_episode``
        (optional) fires as ``(arch_name, record)`` so a UI can show live progress.
        """
        self._require_data()
        episodes = episodes if episodes is not None else self.cfg.training.episodes
        histories: dict[str, list] = {}
        for name, dueling in (("Dueling DQN", True), ("Plain DQN", False)):
            agent = DQNAgent(self._cfg_with_dueling(dueling))
            relay = (lambda record, n=name: on_episode(n, record)) if on_episode else None
            histories[name] = TrainingService(self._env("train"), agent).train(episodes, on_episode=relay)
        return histories

    def recommend(self, split: str = "test") -> dict:
        """Recommend an action for the latest window, assuming a flat portfolio."""
        self._require_data()
        features, _ = self._splits[split]
        window = self.cfg.features.window_size
        market = features.to_numpy(dtype="float32")[-window:]
        return self.inference.recommend(assemble_state(market, position=0.0, cash_exposure=1.0))

    def save_brain(self, path: str) -> None:
        self.agent.save(assert_in_project(path))

    def load_brain(self, path: str) -> None:
        self.agent.load(assert_in_project(path))
