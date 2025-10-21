

import torch


def evaluate_policy(agent,  args, mask_N, envs, test_num_envs):
    agent.eval()
    device = next(agent.parameters()).device
    returns = 0.0
    mask = torch.ones(args.num_users + 1,  (4+args.num_VMs*args.num_times), device=device)
    mask[mask_N, :] = 0
    mask = mask[:-1, :]
    for i in range(test_num_envs):
        terminated = False
        env = envs[i]()
        env.reset(seed=22)
        obs, _ = env.reset()
        j = 0
        while not terminated:

            j = j + 1
            with torch.no_grad():
                obs = torch.Tensor(obs).to(device).unsqueeze(0)
                obs[:, : args.num_users * (4+args.num_VMs*args.num_times)] = obs[:, : args.num_users * (4+args.num_VMs*args.num_times)] * (
                    mask.reshape(-1).unsqueeze(0))


                actions,_ = agent.get_action(obs)
                actions = actions.squeeze(0)


            next_obs, reward, terminated, _, _ = env.step(actions)

            returns += reward
            obs = next_obs
    return returns / test_num_envs




def make_env(env_class, *args):
    def thunk():
        env = env_class(*args)
        return env

    return thunk

