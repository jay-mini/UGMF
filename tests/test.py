import torch

from ugm.objectives.ddpm import DDPMobjective

objective = DDPMobjective(num_timesteps=1000, )

x0 = torch.ones(10000, 1, 1, 1)
noise = torch.randn_like(x0)

for step in [0, 100, 500, 999]:
    t = torch.full((x0.shape[0],), fill_value=step, dtype=torch.long)

    xt = objective.q_sample(x0, t, noise=noise)

    print(
        step,
        "mean:",
        xt.mean().item(),
        "std:",
        xt.std().item(),
    )