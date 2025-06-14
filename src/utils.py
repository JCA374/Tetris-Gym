import os


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_rewards(reward_list, save_path):
    # simple matplotlib plot and save
