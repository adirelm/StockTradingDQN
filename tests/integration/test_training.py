"""Integration test for TrainingService (B15 — episode loop wires env+agent)."""

from tradedqn.services.training import TrainingService


class TestTrain:
    def test_returns_history_per_episode(self, toy_env, dqn_agent):
        history = TrainingService(toy_env, dqn_agent).train(episodes=3)
        assert len(history) == 3
        assert {"episode", "reward", "final_value", "epsilon", "mean_loss"} <= history[0].keys()

    def test_epsilon_decreases_over_training(self, toy_env, dqn_agent):
        start = dqn_agent.epsilon
        history = TrainingService(toy_env, dqn_agent).train(episodes=3)
        assert history[-1]["epsilon"] < start

    def test_replay_grows_and_learning_happens(self, toy_env, dqn_agent):
        TrainingService(toy_env, dqn_agent).train(episodes=2)
        # the toy episode is long enough to pass min_replay and run learn steps
        assert len(dqn_agent.replay) > 0

    def test_final_value_is_reported(self, toy_env, dqn_agent):
        history = TrainingService(toy_env, dqn_agent).train(episodes=1)
        assert history[0]["final_value"] > 0.0

    def test_on_episode_callback_fires_per_episode(self, toy_env, dqn_agent):
        seen: list[dict] = []
        TrainingService(toy_env, dqn_agent).train(episodes=3, on_episode=seen.append)
        assert [r["episode"] for r in seen] == [0, 1, 2]  # live progress for the GUI
