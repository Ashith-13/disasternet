try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
    from models import DisasterAction, DisasterObservation, DisasterEpisodeState

    class DisasterNetEnv(
        EnvClient[DisasterAction, DisasterObservation, DisasterEpisodeState]
    ):
        def _step_payload(self, action):
            return {
                "zone_priorities":  action.zone_priorities,
                "rescue_teams":     action.rescue_teams,
                "medical_units":    action.medical_units,
                "engineering":      action.engineering,
                "comms_restore":    action.comms_restore,
                "helicopter_recon": action.helicopter_recon,
            }

        def _parse_result(self, payload):
            obs = payload.get("observation", {})
            return StepResult(
                observation=DisasterObservation(
                    done=payload.get("done", False),
                    reward=payload.get("reward"),
                    zones=obs.get("zones", []),
                    resources=obs.get("resources", {}),
                    hours_elapsed=obs.get("hours_elapsed", 0.0),
                    lives_saved=obs.get("lives_saved", 0),
                    lives_lost=obs.get("lives_lost", 0),
                    weather=obs.get("weather", "CLEAR"),
                    aftershock_occurred=obs.get("aftershock_occurred", False),
                    hospital_operational=obs.get("hospital_operational", True),
                    comms_coverage=obs.get("comms_coverage", 1.0),
                    cascade_events=obs.get("cascade_events", []),
                    outcome_summary=obs.get("outcome_summary", ""),
                    survival_window=obs.get("survival_window", "HIGH"),
                    task_description=obs.get("task_description", ""),
                ),
                reward=payload.get("reward"),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload):
            return DisasterEpisodeState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                magnitude=payload.get("magnitude", 7.2),
                city_population=payload.get("city_population", 500000),
                total_zones=payload.get("total_zones", 10),
                max_epochs=payload.get("max_epochs", 12),
                task_id=payload.get("task_id", "dynamic_command"),
                scenario_seed=payload.get("scenario_seed", 0),
                baseline_human_score=payload.get("baseline_human_score", 0.71),
            )

except ImportError:
    print("Install openenv-core: pip install openenv-core")
