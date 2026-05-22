"""Tests for InferenceService (B18 — latest window → Q → Buy/Hold/Sell)."""

import numpy as np

from tradedqn.services.inference import InferenceService, action_names

WINDOW, FEATURES = 5, 10


def a_state():
    return np.full((WINDOW, FEATURES), 0.4, dtype=np.float32)


class TestActionNames:
    def test_ordered_by_config_index(self, tiny_cfg):
        assert action_names(tiny_cfg) == ["sell", "hold", "buy"]


class TestRecommend:
    def test_recommendation_shape(self, dqn_agent, tiny_cfg):
        svc = InferenceService.from_config(dqn_agent, tiny_cfg)
        rec = svc.recommend(a_state())
        assert rec["action"] in ("sell", "hold", "buy")
        assert len(rec["q_values"]) == 3

    def test_action_index_is_argmax(self, dqn_agent, tiny_cfg):
        svc = InferenceService.from_config(dqn_agent, tiny_cfg)
        rec = svc.recommend(a_state())
        assert rec["action_index"] == int(np.argmax(rec["q_values"]))
        assert svc.names[rec["action_index"]] == rec["action"]

    def test_matches_agent_greedy_action(self, dqn_agent, tiny_cfg):
        svc = InferenceService.from_config(dqn_agent, tiny_cfg)
        state = a_state()
        assert rec_index(svc, state) == dqn_agent.act(state, greedy=True)

    def test_confidence_and_feature_attribution(self, dqn_agent, tiny_cfg):
        svc = InferenceService.from_config(dqn_agent, tiny_cfg)
        rec = svc.recommend(a_state())
        assert 0.0 <= rec["confidence"] <= 1.0          # softmax probability of the chosen action
        assert len(rec["top_features"]) == 3            # top-k saliency drivers (§8 explanation)
        assert set(rec["top_features"]) <= set(tiny_cfg.features.names)


def rec_index(svc, state):
    return svc.recommend(state)["action_index"]
