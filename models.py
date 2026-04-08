from typing import List, Optional, Dict
from pydantic import Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    from pydantic import BaseModel
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0


class DisasterAction(Action):
    zone_priorities: List[int] = Field(
        default_factory=lambda: list(range(10)),
        description="Zone IDs ranked by priority. Index 0 = highest."
    )
    rescue_teams: Dict[str, int] = Field(
        default_factory=dict,
        description="zone_id -> number of rescue teams. e.g. {'0':3,'2':2}"
    )
    medical_units: Dict[str, int] = Field(
        default_factory=dict,
        description="zone_id -> number of medical units."
    )
    engineering: List[int] = Field(
        default_factory=list,
        description="Zone IDs for road clearing. Send BEFORE rescue teams."
    )
    comms_restore: List[int] = Field(
        default_factory=list,
        description="Zone IDs to restore communications."
    )
    helicopter_recon: List[int] = Field(
        default_factory=list,
        description="Zone IDs for aerial reconnaissance."
    )


class DisasterObservation(Observation):
    # done and reward are INHERITED from Observation — do NOT redefine
    zones: List[dict] = Field(default_factory=list)
    resources: dict = Field(default_factory=dict)
    hours_elapsed: float = Field(default=0.0)
    lives_saved: int = Field(default=0)
    lives_lost: int = Field(default=0)
    weather: str = Field(default="CLEAR")
    aftershock_occurred: bool = Field(default=False)
    hospital_operational: bool = Field(default=True)
    comms_coverage: float = Field(default=1.0)
    cascade_events: List[str] = Field(default_factory=list)
    outcome_summary: str = Field(default="")
    survival_window: str = Field(default="HIGH")
    task_description: str = Field(default="")


class DisasterEpisodeState(State):
    # episode_id and step_count are INHERITED from State — do NOT redefine
    magnitude: float = Field(default=7.2)
    city_population: int = Field(default=500000)
    total_zones: int = Field(default=10)
    max_epochs: int = Field(default=12)
    task_id: str = Field(default="dynamic_command")
    scenario_seed: int = Field(default=0)
    baseline_human_score: float = Field(default=0.71)
