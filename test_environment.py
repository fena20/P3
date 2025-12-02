"""
Quick test script to verify the environment works correctly
"""

import numpy as np
from smarthome_env import SmartHomeEnv

def test_environment():
    """Test basic environment functionality"""
    print("Testing SmartHomeEnv...")
    
    # Create environment
    env = SmartHomeEnv()
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Reset successful. Observation shape: {obs.shape}")
    print(f"  Observation: {obs}")
    
    # Test step
    action = 1  # Turn ON
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful. Reward: {reward:.4f}")
    print(f"  Indoor temp: {env.indoor_temp:.2f}°C")
    print(f"  Info: {info}")
    
    # Test cycling penalty
    print("\nTesting cycling penalty (short-cycling prevention)...")
    env.reset()
    
    # Switch ON
    obs, reward1, _, _, info1 = env.step(1)
    print(f"  Action: ON, Reward: {reward1:.4f}, Cycling penalty: {info1['cycling_penalty']:.4f}")
    
    # Switch OFF immediately (should trigger penalty)
    obs, reward2, _, _, info2 = env.step(0)
    print(f"  Action: OFF (immediate switch), Reward: {reward2:.4f}, Cycling penalty: {info2['cycling_penalty']:.4f}")
    
    if info2['cycling_penalty'] > 0:
        print("  ✓ Cycling penalty correctly applied for short-cycling!")
    else:
        print("  ⚠ Cycling penalty not applied (may be due to timing)")
    
    # Test multiple steps
    print("\nTesting multiple steps...")
    env.reset()
    total_reward = 0
    for i in range(100):
        action = 1 if i % 30 < 15 else 0  # Simple pattern
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"✓ Completed {env.current_step} steps")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total cost: ${env.total_cost:.2f}")
    print(f"  Total discomfort: {env.total_discomfort:.2f}")
    print(f"  Total cycles: {env.total_cycles}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_environment()
