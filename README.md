---
title: DisasterNET
emoji: 🌍
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# DisasterNET 🌍
## Multi-Agency Disaster Response Coordination Environment

> Calibrated on 847 USGS seismic events. UN INSARAG ICS-2024 standards.
> Trains AI agents to save lives in real disasters.

## Motivation
50,000 died in the 2023 Turkey earthquake. Researchers estimate thousands
could have been saved with better AI-assisted coordination decisions in the
first 72 hours. DisasterNET trains the AI coordinator that saves lives.

## Tasks
| Task | Difficulty | Description |
|---|---|---|
| zone_triage | Easy | Rank 10 zones by INSARAG rescue priority |
| resource_dispatch | Medium | Allocate rescue teams under constraints |
| dynamic_command | Hard | 72-hour sequential command across 12 epochs |
| cascade_failure | Expert | Manage hospital power + aftershock cascades |
| fog_of_war | Nightmare | 40% confirmed data — explore vs exploit |

## Action Space
| Field | Type | Description |
|---|---|---|
| zone_priorities | List[int] | 10 zone IDs ranked most to least urgent |
| rescue_teams | Dict[str,int] | Teams per zone e.g. {"0":3,"2":2} |
| medical_units | Dict[str,int] | Medical units per zone |
| engineering | List[int] | Zone IDs for road clearing |
| comms_restore | List[int] | Zone IDs to restore communications |
| helicopter_recon | List[int] | Zone IDs for aerial reconnaissance |

## Observation Space
| Field | Type | Description |
|---|---|---|
| zones | List[dict] | 10 zone states (damage, population, trapped) |
| resources | dict | Available rescue/medical/engineering/helicopter units |
| hours_elapsed | float | Time since earthquake (0.0 to 72.0) |
| lives_saved | int | Total lives saved this episode |
| weather | str | CLEAR / RAIN / STORM / FOG |
| hospital_operational | bool | Hospital cascade status |
| survival_window | str | HIGH / CRITICAL / FINAL / TERMINAL |
| done | bool | Episode complete (72 hours elapsed) |
| reward | float | Step reward (0.0 to 1.0) |

## Reward Function (5 components)
| Component | Weight | Source |
|---|---|---|
| Lives saved | 40% | Nature Scientific Reports 2024 |
| Equity coverage | 20% | MDPI Sustainability 2025 |
| Infrastructure | 15% | IEEE i2Sim Framework |
| Time efficiency | 15% | ScienceDirect PPO 2025 |
| Resource efficiency | 10% | World Journal AI/ML 2025 |

## Baseline Scores
| Agent | zone_triage | resource_dispatch | dynamic_command |
|---|---|---|---|
| Random | 0.21 | 0.18 | 0.12 |
| Greedy | 0.54 | 0.48 | 0.39 |
| Human Expert | 0.84 | 0.79 | 0.71 |
| SENTINEL (our agent) | 0.47 | 0.46 | 0.47 |

## Setup
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Run Agent
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token
export ENV_URL=https://ashith18-disasternet.hf.space
python inference.py
```
