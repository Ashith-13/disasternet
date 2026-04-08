import random
import uuid
import math
import sys
import os
import numpy as np


from models import DisasterAction, DisasterObservation, DisasterEpisodeState

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass


class DisasterNetEnvironment(Environment):
    """
DisasterNET: Multi-Agency Disaster Response Coordination
A real-world disaster response simulation for evaluating AI agents
on triage, resource allocation, and multi-step emergency decision-making.
"""

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_EPOCHS = 12

    SURVIVAL_CURVE = {
        (0,  24): 0.90,
        (24, 48): 0.55,
        (48, 72): 0.22,
        (72, 999): 0.05,
    }

    ZONE_TYPES = [
        "HOSPITAL", "RESIDENTIAL", "SCHOOL", "RESIDENTIAL",
        "INDUSTRIAL", "RESIDENTIAL", "GOVERNMENT", "RESIDENTIAL",
        "BRIDGE", "RESIDENTIAL"
    ]

    TASK_DESCRIPTIONS = {
        "zone_triage": (
            "TASK 1 (Easy): Rank all 10 zones by rescue priority. "
            "Use INSARAG ICS methodology based on damage, population, zone type."
        ),
        "resource_dispatch": (
            "TASK 2 (Medium): Allocate rescue_teams and medical_units. "
            "Check road_access before sending teams. Protect hospital Zone 0."
        ),
        "dynamic_command": (
            "TASK 3 (Hard): 72-hour sequential command across 12 epochs. "
            "Adapt to aftershocks, weather, hospital cascade failures."
        ),
        "cascade_failure": (
            "TASK 4 (Expert): Hospital loses power at Hour 12 if engineering "
            "not sent to Zone 0. Aftershock collapses zones mid-episode."
        ),
        "fog_of_war": (
            "TASK 5 (Nightmare): Only 40% of zone data confirmed. "
            "Decide: gather information vs immediate rescue."
        ),
    }

    def __init__(self):
        self._state = DisasterEpisodeState()
        self._zones = []
        self._resources = {}
        self._hours = 0.0
        self._lives_saved = 0
        self._lives_lost = 0
        self._cascade_events = []
        self._weather = "CLEAR"
        self._aftershock_prob = 0.15
        self._hospital_operational = True
        self._hospital_backup_hours = 12.0
        self._comms_coverage = 1.0
        self._task_id = "dynamic_command"
        self._seed = 0
        self._fog_of_war = False

    def reset(self, seed=None, episode_id=None, task_id=None, **kwargs):
        self._seed = seed if seed is not None else random.randint(0, 99999)
        random.seed(self._seed)
        np.random.seed(self._seed % (2 ** 31))

        self._task_id = task_id or "dynamic_command"
        self._fog_of_war = (self._task_id == "fog_of_war")
        self._hours = 0.0
        self._lives_saved = 0
        self._lives_lost = 0
        self._cascade_events = []
        self._weather = "CLEAR"
        self._aftershock_prob = 0.15
        self._hospital_operational = True
        self._hospital_backup_hours = 12.0
        self._comms_coverage = 1.0

        magnitude = round(random.uniform(6.5, 8.5), 1)
        self._zones = self._generate_zones(magnitude)
        self._resources = self._generate_resources(magnitude)

        if self._fog_of_war:
            self._zones = self._apply_fog_of_war(self._zones)

        self._state = DisasterEpisodeState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            magnitude=magnitude,
            city_population=500000,
            total_zones=10,
            max_epochs=self.MAX_EPOCHS,
            task_id=self._task_id,
            scenario_seed=self._seed,
            baseline_human_score=0.71,
        )

        return DisasterObservation(
            done=False,
            reward=None,
            zones=self._zones,
            resources=self._resources,
            hours_elapsed=0.0,
            lives_saved=0,
            lives_lost=0,
            weather="CLEAR",
            aftershock_occurred=False,
            hospital_operational=True,
            comms_coverage=1.0,
            cascade_events=[],
            outcome_summary=(
                f"M{magnitude} earthquake struck city of 500,000. "
                f"72-hour golden window begins. Survival: 90%. Respond NOW."
            ),
            survival_window="HIGH",
            task_description=self.TASK_DESCRIPTIONS.get(
                self._task_id, self.TASK_DESCRIPTIONS["dynamic_command"]
            ),
        )

    def step(self, action, timeout_s=None, **kwargs):
        self._state.step_count += 1
        self._hours += 6.0
        cascade = []

        # Aftershock
        aftershock = random.random() < self._aftershock_prob
        if aftershock:
            mag = round(random.uniform(4.0, 5.5), 1)
            for zone in self._zones:
                if zone["aftershock_risk"] > 0.5:
                    zone["damage_level"] = min(
                        1.0, zone["damage_level"] + random.uniform(0.05, 0.15)
                    )
                    zone["trapped_confirmed"] += random.randint(50, 200)
            cascade.append(
                f"Hour{self._hours:.0f}: Aftershock M{mag} — high-risk zones worsened"
            )
            self._aftershock_prob *= 0.7

        # Weather
        self._weather = self._get_weather(self._hours)
        helicopters_grounded = self._weather in ["STORM", "FOG"]

        # Hospital cascade
        self._hospital_backup_hours -= 6.0
        if self._hospital_backup_hours <= 0 and self._hospital_operational:
            if 0 not in action.engineering:
                self._hospital_operational = False
                cascade.append(
                    f"Hour{self._hours:.0f}: CRITICAL — Hospital power FAILED. "
                    f"Medical units -40% effectiveness."
                )

        # Simulate response
        outcome = self._simulate_response(action, cascade, helicopters_grounded)
        self._lives_saved += outcome["lives_saved"]
        self._lives_lost += outcome["lives_lost"]
        self._cascade_events.extend(cascade)
        self._consume_resources(action)

        # Update comms
        for z_id in action.comms_restore:
            if 0 <= z_id < len(self._zones):
                self._zones[z_id]["comms_available"] = True

        if not helicopters_grounded:
            for z_id in action.helicopter_recon:
                if 0 <= z_id < len(self._zones):
                    self._zones[z_id]["comms_available"] = True
                    self._zones[z_id]["trapped_estimated"] = (
                        self._zones[z_id]["trapped_confirmed"]
                    )

        zones_with_comms = sum(1 for z in self._zones if z["comms_available"])
        self._comms_coverage = zones_with_comms / len(self._zones)

        reward = self._compute_reward(action, outcome)
        done = (self._hours >= 72.0 or self._state.step_count >= self.MAX_EPOCHS)

        return DisasterObservation(
            done=done,
            reward=reward,
            zones=self._zones,
            resources=self._resources,
            hours_elapsed=self._hours,
            lives_saved=self._lives_saved,
            lives_lost=self._lives_lost,
            weather=self._weather,
            aftershock_occurred=aftershock,
            hospital_operational=self._hospital_operational,
            comms_coverage=self._comms_coverage,
            cascade_events=cascade,
            outcome_summary=outcome["summary"],
            survival_window=self._get_survival_window(self._hours),
            task_description=self.TASK_DESCRIPTIONS.get(
                self._task_id, self.TASK_DESCRIPTIONS["dynamic_command"]
            ),
        )

    @property
    def state(self):
        return self._state

    def _generate_zones(self, magnitude):
        zones = []
        for i, zone_type in enumerate(self.ZONE_TYPES):
            base_damage = (magnitude - 5.0) / 4.0
            damage = min(1.0, base_damage * random.uniform(0.4, 1.3))
            damage = round(damage, 3)
            if zone_type == "HOSPITAL":
                population = random.randint(500, 2000)
            elif zone_type == "SCHOOL":
                population = random.randint(200, 800)
            elif zone_type == "INDUSTRIAL":
                population = random.randint(100, 500)
            else:
                population = random.randint(10000, 80000)
            trapped = int(population * damage * random.uniform(0.02, 0.08))
            trapped = max(0, trapped)
            zones.append({
                "zone_id": i,
                "zone_type": zone_type,
                "damage_level": damage,
                "population": population,
                "trapped_confirmed": trapped,
                "trapped_estimated": int(trapped * random.uniform(0.6, 1.4)),
                "road_access": round(random.uniform(max(0.05, 1.0 - damage), 1.0), 2),
                "has_power": random.random() > (damage * 0.7),
                "comms_available": random.random() > (damage * 0.5),
                "aftershock_risk": round(random.uniform(0.1, 0.9), 2),
            })
        return zones

    def _apply_fog_of_war(self, zones):
        for zone in zones:
            if random.random() > 0.4:
                zone["trapped_confirmed"] = 0
                zone["comms_available"] = False
                zone["data_quality"] = "ESTIMATED"
            else:
                zone["data_quality"] = "CONFIRMED"
        return zones

    def _generate_resources(self, magnitude):
        scale = max(0.3, 1.0 - (magnitude - 6.0) / 5.0)
        return {
            "rescue_teams":      int(15 * scale),
            "medical_units":     int(10 * scale),
            "food_water_tons":   round(500.0 * scale, 1),
            "helicopters":       int(5 * scale),
            "engineering_crews": int(8 * scale),
            "comms_units":       int(6 * scale),
        }

    def _simulate_response(self, action, cascade, helicopters_grounded):
        total_saved = 0
        total_lost = 0
        survival_prob = self._get_survival_prob(self._hours)

        for i, zone in enumerate(self._zones):
            zone_id = str(i)
            trapped = zone["trapped_confirmed"]
            if trapped <= 0:
                continue

            road_ok = zone["road_access"] > 0.3 or i in action.engineering
            teams = action.rescue_teams.get(zone_id, 0)
            medical = action.medical_units.get(zone_id, 0)

            if teams > 0 and not road_ok:
                cascade.append(
                    f"Zone{i}: {teams} teams BLOCKED — send engineering first"
                )
                teams = 0

            if teams > 0:
                eff = min(1.0, teams / 3.0)
                saved = int(trapped * survival_prob * eff * 0.6)
                total_saved += saved
                zone["trapped_confirmed"] = max(0, trapped - saved)

            if medical > 0 and self._hospital_operational:
                total_saved += int(total_saved * 0.2 * min(1.0, medical / 2.0))
            elif medical > 0:
                total_saved += int(total_saved * 0.08 * min(1.0, medical / 2.0))

            if teams == 0 and trapped > 50:
                total_lost += int(trapped * (1.0 - survival_prob) * 0.08)

        summary = (
            f"Hour {self._hours:.0f}/72 | "
            f"Saved: +{total_saved} | Lost: +{total_lost} | "
            f"Total: {self._lives_saved + total_saved} | "
            f"Hospital: {'OK' if self._hospital_operational else 'OFFLINE'}"
        )
        return {"lives_saved": total_saved, "lives_lost": total_lost, "summary": summary}

    def _compute_reward(self, action, outcome):
        total_trapped = sum(z["trapped_confirmed"] for z in self._zones)
        if total_trapped <= 0:
            return 1.0
        survival_prob = self._get_survival_prob(self._hours)
        max_saveable = total_trapped * survival_prob
        lives_score = min(1.0, outcome["lives_saved"] / max(max_saveable, 1))

        zones_responded = sum(
            1 for i in range(len(self._zones))
            if str(i) in action.rescue_teams or
               str(i) in action.medical_units or
               i in action.engineering
        )
        equity_score = zones_responded / len(self._zones)

        avg_road = sum(z["road_access"] for z in self._zones) / len(self._zones)
        infra_score = (
            0.40 * float(self._hospital_operational) +
            0.30 * self._comms_coverage +
            0.30 * avg_road
        )
        time_score = math.exp(-0.03 * self._hours) * min(1.0, max(0.0, lives_score))

        total_dispatched = sum(action.rescue_teams.values())
        wasted = sum(
            t for z_id, t in action.rescue_teams.items()
            if int(z_id) < len(self._zones) and
               self._zones[int(z_id)]["road_access"] < 0.3 and
               int(z_id) not in action.engineering
        )
        efficiency = 1.0 - min(1.0, wasted / max(total_dispatched, 1))

        reward = (
            0.40 * lives_score +
            0.20 * equity_score +
            0.15 * infra_score +
            0.15 * time_score +
            0.10 * efficiency
        )
        return float(np.clip(reward, 0.0, 1.0))

    def _consume_resources(self, action):
        r = self._resources
        r["rescue_teams"]      = max(0, r["rescue_teams"]      - sum(action.rescue_teams.values()) // 3)
        r["medical_units"]     = max(0, r["medical_units"]     - sum(action.medical_units.values()) // 4)
        r["engineering_crews"] = max(0, r["engineering_crews"] - len(action.engineering) // 2)
        r["helicopters"]       = max(0, r["helicopters"]       - len(action.helicopter_recon) // 3)
        r["food_water_tons"]   = max(0.0, r["food_water_tons"] - 20.0)
        if 35 <= self._hours <= 37:
            r["rescue_teams"]    += 5
            r["medical_units"]   += 3
            r["food_water_tons"] += 100.0

    def _get_weather(self, hours):
        if hours < 18:   return "CLEAR"
        elif hours < 30: return random.choice(["CLEAR", "CLEAR", "RAIN"])
        elif hours < 48: return random.choice(["RAIN", "RAIN", "STORM"])
        else:            return random.choice(["RAIN", "STORM", "FOG"])

    def _get_survival_prob(self, hours):
        for (low, high), prob in self.SURVIVAL_CURVE.items():
            if low <= hours < high:
                return prob
        return 0.05

    def _get_survival_window(self, hours):
        if hours < 24:   return "HIGH"
        elif hours < 48: return "CRITICAL"
        elif hours < 72: return "FINAL"
        else:            return "TERMINAL"
