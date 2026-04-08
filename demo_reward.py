import sys
import os
import yaml
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.environment import ToxiCleanEnv
    print("✅ Environment logic imported successfully.")
except ImportError as e:
    print(f"❌ Failed to import environment: {e}")
    sys.exit(1)

def run_headless_demo():
    print("--- ToxiClean AI: Headless Moderation Demo ---")
    
    # Initialize environment with Task 1 (Spam) - Deterministic
    env = ToxiCleanEnv(task_name="spam_detection")
    obs = env.reset()
    
    print(f"\n[Observation]: {obs.content[:100]}...")
    print(f"[Metadata]: Task={obs.task_name}, Step={obs.step_index}")
    
    # Simulate a "DELETE" action (Action for Spam/Toxicity)
    action = "DELETE"
    next_obs, reward, done, info = env.step(action)
    
    print("\n--- Moderation Result ---")
    print(f"Action Taken: {'REJECT' if action == 0 else 'APPROVE'}")
    print(f"Reward Received: {reward}")
    print(f"Grader Reasoning: {info.get('reasoning', 'No reasoning provided.')}")
    print(f"Normalized Score: {info.get('normalized_performance', 0.0)}")
    print("--------------------------")

if __name__ == "__main__":
    run_headless_demo()
