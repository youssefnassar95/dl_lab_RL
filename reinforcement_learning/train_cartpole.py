import sys
sys.path.append("../")
from utils import EpisodeStats  # noqa: E402
from agent.networks import MLP  # noqa: E402
from tensorboard_evaluation import *  # noqa: E402
from agent.dqn_agent import DQNAgent  # noqa: E402
import itertools as it  # noqa: E402
import gym  # noqa: E402
import numpy as np  # noqa: E402


def run_episode(env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()        # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, model_dir="./models_cartpole", tensorboard_dir="./tensorboard"):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(tensorboard_dir, "lorem", stats=[
                             "episode_reward", "a_0", "a_1"])

    # training
    for i in range(num_episodes):

        stats = run_episode(env, agent, deterministic=False, do_training=True)
        print("episode: ", i)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        stats_eval = run_episode(
            env, agent, deterministic=True, do_training=False)
        score = []
        if i % eval_cycle == 0:
            for j in range(num_eval_episodes):
                stats_eval = run_episode(
                    env, agent, deterministic=True, do_training=False)
                score.append(stats_eval.episode_reward)
            print("episode: ", i, sum(score)/len(score))
            tensorboard.write_episode_data(i, eval_dict={"episode_reward": sum(score)/len(score),
                                                         "a_0": stats.get_action_usage(0),
                                                         "a_1": stats.get_action_usage(1)})

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent.pt"))

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5   # evaluate on 5 episodes
    eval_cycle = 20         # evaluate every 10 episodes

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    q_network = MLP(state_dim, num_actions)
    target_network = MLP(state_dim, num_actions)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    agent = DQNAgent(q_network, target_network, num_actions)
    # 3. train DQN agent with train_online(...)
    train_online(env, agent, 250)
