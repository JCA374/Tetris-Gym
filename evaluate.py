from config import *
from src.env import make_env
from src.agent import Agent


def main():
    env = make_env(ENV_NAME)
    agent = Agent(env.observation_space, env.action_space)
    agent.load_checkpoint(latest=True)
    for _ in range(10):
        state, done = env.reset(), False
        total_reward = 0
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        print("Episode reward:", total_reward)


if __name__ == "__main__":
    main()
