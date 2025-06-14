from config import *
from src.env import make_env
from src.agent import Agent


def main():
    env = make_env(ENV_NAME)
    agent = Agent(env.observation_space, env.action_space)
    for ep in range(MAX_EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
        agent.save_checkpoint(ep)


if __name__ == "__main__":
    main()
