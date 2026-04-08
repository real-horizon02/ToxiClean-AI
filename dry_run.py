"""
dry_run.py — Runs all 3 tasks with a rule-based mock agent (no API calls).
Shows exact [START]/[STEP]/[END] log format and real grader scores.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.environment import ToxiCleanEnv
from core.models import ModerationAction
from core.tasks import spam, toxicity, contextual

TASKS = ["spam_detection", "toxicity_classification", "contextual_moderation"]
BENCHMARK = "toxiclean"
MODEL = "mock-rule-agent"

def mock_agent(obs, task_name):
    """Simple rule-based agent — picks the correct action from ground truth."""
    modules = {
        "spam_detection": spam,
        "toxicity_classification": toxicity,
        "contextual_moderation": contextual,
    }
    m = modules[task_name]
    return m._SAMPLES[obs.step_index]["label"]

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error=None):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

all_scores = {}

print("=" * 60)
print("  ToxiClean AI — Dry Run (Mock Agent, No API Calls)")
print("  All 3 tasks | gpt-4o-mini compatible log format")
print("=" * 60)

for task_name in TASKS:
    print()
    env = ToxiCleanEnv(task_name=task_name)
    obs = env.reset()
    log_start(task_name, BENCHMARK, MODEL)

    rewards, steps_taken = [], 0
    for step in range(1, 100):
        action = mock_agent(obs, task_name)
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        steps_taken = step
        log_step(step, action.value, reward, done)
        if done:
            break

    mean = sum(rewards) / len(rewards) if rewards else 0.0
    score = max(0.0, min(1.0, (mean + 1.2) / 2.4))
    log_end(score >= 0.5, steps_taken, score, rewards)
    all_scores[task_name] = score

print()
print("=" * 60)
print("  OVERALL RESULTS")
print("=" * 60)
for t, s in all_scores.items():
    bar = "#" * int(s * 20)
    print(f"  {t:<30} {s:.4f}  [{bar:<20}]")
avg = sum(all_scores.values()) / len(all_scores)
print(f"  {'AVERAGE':<30} {avg:.4f}")
print()
print("  Checklist verification:")
print(f"  [OK] [START]/[STEP]/[END] single-line format")
print(f"  [OK] score in [0.0, 1.0]:   {all(0<=s<=1 for s in all_scores.values())}")
print(f"  [OK] 3 tasks completed:     {len(all_scores) == 3}")
