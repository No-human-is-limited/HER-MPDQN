import os
import click
import time
from common import ClickPythonLiteralOption
import gym
import gym_hybrid
from gym.wrappers import Monitor
import numpy as np


def pad_action(act, all_act_param):
    act = (act,)
    param = ([all_act_param[0], all_act_param[1]],)
    action = act + param
    return action


def evaluate(env, agent, episodes=1000):
    count = 0.
    total_reward = 0.
    timestep = 0.
    for i in range(episodes):
        state = env.reset()
        obs = state['observation']
        g = state['desired_goal']
        g = np.array(g, dtype=np.float32, copy=False)
        terminal = False
        while not terminal:
            timestep += 1
            obs = np.array(obs, dtype=np.float32, copy=False)
            input_tensor = agent.preproc_inputs(obs,g)
            act, all_action_parameters = agent.act_eval(input_tensor)
            action = pad_action(act, all_action_parameters)
            state_new, reward, terminal, info = env.step(action)
            obs = state_new['observation']
            total_reward += reward
            if reward == 0:
                count += 1
    return total_reward / episodes, timestep / episodes, count / episodes


def make_env(seed):
    env = gym.make('Moving-v0', seed=seed, penalty=0, max_step=100, use_goal_switch=False)
    return env


@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=30000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--load-model', default=0, help='load_model', type=int)
@click.option('--batch-size', default=128, help='Minibatch size.', type=int)
@click.option('--max-step', default=100, help='max step.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=0, help='Number of transitions required to start learning.',
              type=int)
@click.option('--use-ornstein-noise', default=False,
              help='Use Ornstein noise instead of epsilon-greedy with uniform random exploration.', type=bool)
@click.option('--replay-memory-size', default=150000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.05, help='Final epsilon value.', type=float)
@click.option('--tau-actor', default=0.001, help='Soft target network update averaging factor.', type=float)
@click.option('--tau-actor-param', default=0.001, help='Soft target network update averaging factor.',
              type=float)  # 0.001
@click.option('--learning-rate-actor', default=0.001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--multipass', default=False, help='Separate action-parameter inputs using multiple Q-network passes.',
              type=bool)
@click.option('--indexed', default=False, help='Indexed loss function.', type=bool)
@click.option('--weighted', default=False, help='Naive weighted loss function.', type=bool)
@click.option('--average', default=False, help='Average weighted loss function.', type=bool)
@click.option('--random-weighted', default=False, help='Randomly weighted loss function.', type=bool)
@click.option('--zero-index-gradients', default=False,
              help="Whether to zero all gradients for action-parameters not corresponding to the chosen action.",
              type=bool)
@click.option('--action-input-layer', default=0,
              help='Which layer to input action parameters at when using split Q-networks.', type=int)
@click.option('--layers', default="[256,128,64]", help='Duplicate action-parameter inputs.',
              cls=ClickPythonLiteralOption)
@click.option('--save-freq', default=10000, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="new_results/complex_uav_task", help='Output directory.', type=str)
@click.option('--title', default="PA_DQN_HER_NEW", help="Prefix of output files", type=str)
@click.option('--use-her', default=True, help="her", type=bool)
@click.option('--use_pro', default=False, help="Render game states. Incompatible with save-frames.", type=bool)
@click.option('--alpha', default=0.3, help='useless.', type=float)
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, tau_actor, tau_actor_param, use_ornstein_noise, learning_rate_actor, learning_rate_actor_param,
        title, epsilon_final, clip_grad, scale_actions, indexed, zero_index_gradients, action_input_layer,
        evaluation_episodes, multipass, weighted, average, random_weighted, update_ratio,
        save_freq, save_dir, layers,use_her,load_model,max_step,use_pro,alpha):
    env = make_env(seed)
    dir = os.path.join(save_dir, title)
    env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    np.random.seed(seed)

    from agents.pdqn_her_hybrid import PDQNAgent
    from agents.pdqn_multipass_her_hybrid import MultiPassPDQNAgent
    agent_class = PDQNAgent
    if multipass:
        agent_class = MultiPassPDQNAgent
    agent = agent_class(
        env,
        env.observation_space, env.action_space,
        actor_kwargs={"hidden_layers": layers,
                      'action_input_layer': action_input_layer,
                      'activation': "leaky_relu",
                      'output_layer_init_std': 0.01},
        actor_param_kwargs={"hidden_layers": layers,
                            'activation': "leaky_relu",
                            'output_layer_init_std': 0.01},
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,  # 0.99
        tau_actor=tau_actor,
        tau_actor_param=tau_actor_param,
        clip_grad=clip_grad,
        indexed=indexed,
        weighted=weighted,
        average=average,
        random_weighted=random_weighted,
        initial_memory_threshold=initial_memory_threshold,
        use_ornstein_noise=use_ornstein_noise,
        replay_memory_size=replay_memory_size,
        inverting_gradients=inverting_gradients,
        zero_index_gradients=zero_index_gradients,
        seed=seed,
        alpha=alpha,
        use_pro=use_pro
        )
    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)

    returns = []
    acc = []
    avg_acc = []
    timesteps = []
    avg_step = []
    start_time_train = time.time()
    for i in range(episodes):
        observation = env.reset()
        obs = observation['observation']
        ag = observation['achieved_goal']
        g = observation['desired_goal']
        obs = np.array(obs, dtype=np.float32, copy=False)
        g = np.array(g, dtype=np.float32, copy=False)
        agent.start_episode()
        episode_trans = []
        get_return = 0
        for j in range(max_step):
            input_tensor = agent.preproc_inputs(obs, g)
            act, all_action_parameters = agent.act(input_tensor)
            action = pad_action(act, all_action_parameters)
            next_obs, reward, done, info = env.step(action)
            obs_new = next_obs['observation']
            ag_new = next_obs['achieved_goal']
            if reward == 0 and done:
                get_return = 1
            episode_trans.append([obs, np.concatenate(([act], all_action_parameters)).ravel(), reward, obs_new,
                                  done, g, ag, ag_new])
            ag = ag_new
            obs = np.array(obs_new, dtype=np.float32, copy=False)
            if done:
                break

        if use_her:
            agent.save_episode(episode_trans, reward_func=env.compute_reward)

        for _ in range(int(j*update_ratio)+1):
            agent.optimize_td_loss()
        agent.end_episode()
        returns.append(get_return)
        timesteps.append(j+1)
        if (i+1) % 100 == 0:
            print('{0:5s} r100:{1:.4f} success rate:{2:.4f}'.format(str(i + 1), np.array(returns[-100:]).mean(),
                                                                    (np.array(returns) == 1.).sum() / len(
                                                                        returns)))
            avg_acc.append(np.array(returns[-100:]).mean())
            acc.append((np.array(returns) == 1.).sum() / len(returns))
            avg_step.append(np.array(timesteps[-100:]).mean())

        if save_freq > 0 and save_dir and (i + 1) % save_freq == 0:
            agent.save_models(os.path.join(os.path.join(dir, str(seed)), str(i + 1)))
    end_time_train = time.time()
    np.save(os.path.join(os.path.join(dir, str(seed)), title + "avg_step"), avg_step)
    np.save(os.path.join(os.path.join(dir, str(seed)), title + "acc"), acc)
    np.save(os.path.join(os.path.join(dir, str(seed)), title + "avg_acc"), avg_acc)
    # if evaluation_episodes > 0:
    #     print("Evaluating agent over {} episodes".format(evaluation_episodes))
    #     agent.epsilon_final = 0
    #     agent.epsilon = 0
    #     agent.noise = None
    #     agent.actor.eval()
    #     agent.actor_param.eval()
    #     evaluation_returns, evaluation_steps, success_rate = evaluate(env, agent, evaluation_episodes)
    #     np.save(os.path.join(os.path.join(dir, str(seed)), title + "eval_return"), evaluation_returns)
    #     np.save(os.path.join(os.path.join(dir, str(seed)), title + "eval_step"), evaluation_steps)
    #     np.save(os.path.join(os.path.join(dir, str(seed)), title + "eval_success_rate"), success_rate)
    print("Training time: %.2f seconds" % (end_time_train - start_time_train))
    env.close()


if __name__ == '__main__':
    run()
