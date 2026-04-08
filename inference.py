#!/usr/bin/env python3
import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

# ── Environment variables ────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("DISASTERNET_TASK",      "dynamic_command")
BENCHMARK    = os.getenv("DISASTERNET_BENCHMARK", "disasternet")

# ── YOUR HF SPACE URL ────────────────────────────────────────
# Already deployed at: https://huggingface.co/spaces/Ashith18/disasternet
# The actual Space URL for API calls:
ENV_URL = os.getenv("ENV_URL", "https://ashith18-disasternet.hf.space")

MAX_STEPS         = 8
MAX_TOKENS        = 500
TEMPERATURE       = 0.1
SUCCESS_THRESHOLD = 0.30

# ── EXACT LOG FORMAT (from official sample) ──────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ── SENTINEL System Prompt ───────────────────────────────────
SENTINEL_SYSTEM = textwrap.dedent("""
You are SENTINEL — expert AI disaster response coordinator.
RULES:
1. HOSPITAL FIRST: Always send engineering to Zone 0 in epoch 1.
2. ROADS BEFORE TEAMS: Never send rescue_teams to zone with road_access < 0.3 unless also in engineering list.
3. RESERVE: Keep 2 helicopters for aftershocks.
4. EQUITY: Respond to 7+ zones for equity bonus.

Return ONLY valid JSON:
{
  "zone_priorities": [list of 10 zone_ids ranked most to least urgent],
  "rescue_teams": {"zone_id": count},
  "medical_units": {"zone_id": count},
  "engineering": [zone_id_list],
  "comms_restore": [zone_id_list],
  "helicopter_recon": [zone_id_list]
}
No explanation. Just JSON.
""").strip()

SENTINEL_LESSONS = []

def get_action(client, obs, step, history):
    zones = obs.get("zones", [])
    zone_lines = "\n".join(
        f"  Zone{z['zone_id']} [{z['zone_type']}] "
        f"dmg={z.get('damage_level',0):.0%} "
        f"trapped={z.get('trapped_confirmed',0)} "
        f"road={'CLEAR' if z.get('road_access',1)>0.3 else 'BLOCKED'}"
        for z in zones
    )
    lessons = "\n".join(f"- {l}" for l in SENTINEL_LESSONS[-5:]) or "None yet"
    user = f"""
Hour {obs.get('hours_elapsed',0):.0f}/72 | Window: {obs.get('survival_window','HIGH')} | Weather: {obs.get('weather','CLEAR')}
Hospital: {'ONLINE' if obs.get('hospital_operational',True) else 'OFFLINE'}
Lives saved: {obs.get('lives_saved',0)} | Resources: {json.dumps(obs.get('resources',{}),default=str)}
Zones:
{zone_lines}
Cascades: {obs.get('cascade_events',[])[-2:]}
Lessons: {lessons}
History: {history[-3:]}
Output JSON action:"""
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SENTINEL_SYSTEM},
                {"role": "user",   "content": user}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=30,
        )
        text = r.choices[0].message.content.strip()
        s, e = text.find('{'), text.rfind('}') + 1
        if s >= 0 and e > s:
            d = json.loads(text[s:e])
            for k in ["zone_priorities","rescue_teams","medical_units","engineering","comms_restore","helicopter_recon"]:
                if k not in d:
                    d[k] = [] if k not in ["rescue_teams","medical_units"] else {}
            return d
    except Exception as ex:
        print(f"[DEBUG] Agent error: {ex}", flush=True)
    # Safe fallback
    return {
        "zone_priorities":  [0,2,6,1,3,5,7,9,4,8],
        "rescue_teams":     {"2":3,"1":2},
        "medical_units":    {"2":2,"1":1},
        "engineering":      [0,5],
        "comms_restore":    [7],
        "helicopter_recon": [4,9],
    }

def reflect(score, lives, history):
    global SENTINEL_LESSONS
    if score > 0.6:
        SENTINEL_LESSONS.append(f"Score {score:.2f} — saved {lives} lives — strategy effective")
    else:
        SENTINEL_LESSONS.append(f"Score {score:.2f} — prioritize hospital + road clearing earlier")
    SENTINEL_LESSONS = SENTINEL_LESSONS[-8:]

async def run_task(env, client, task_id):
    rewards, steps_taken, score, success = [], 0, 0.0, False
    history, final_lives = [], 0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment with task
        result = await env.reset(task_id=task_id)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            obs_dict = result.observation.model_dump()
            raw = get_action(client, obs_dict, step, history)

            try:
                from models import DisasterAction
                action = DisasterAction(**raw)
            except Exception as e:
                print(f"[DEBUG] Action error: {e}", flush=True)
                from models import DisasterAction
                action = DisasterAction(
                    zone_priorities=list(range(10)),
                    rescue_teams={"0":2,"2":2},
                    medical_units={"2":1},
                    engineering=[0],
                )

            result = await env.step(action)
            reward = result.reward or 0.0
            done   = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(raw)[:120].replace("\n"," "),
                reward=reward,
                done=done,
                error=None,
            )

            obs = result.observation
            history.append(f"Step{step}: r={reward:.2f} lives={obs.lives_saved}")
            final_lives = obs.lives_saved

            if done:
                break

        score   = min(max(sum(rewards) / (MAX_STEPS * 1.0), 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
        reflect(score, final_lives, history)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] close error: {e}", flush=True)
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    from client import DisasterNetEnv

    # Connect directly to your deployed HF Space
    # No Docker needed — uses WebSocket connection
    print(f"[DEBUG] Connecting to: {ENV_URL}", flush=True)

    async with DisasterNetEnv(base_url=ENV_URL) as env:
        tasks = ["zone_triage", "resource_dispatch", "dynamic_command"]
        scores = {}

        for task_id in tasks:
            scores[task_id] = await run_task(env, client, task_id)
            print(
                f"[DEBUG] {task_id}: {scores[task_id]:.3f} "
                f"(human baseline=0.710)",
                flush=True
            )

        avg = sum(scores.values()) / len(scores)
        print(f"[DEBUG] SENTINEL avg={avg:.3f} vs human=0.710", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
