from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from ugm.models.unet import SimpleUNet
from ugm.objectives.ddpm import DDPMobjective

def seed_everything(seed: int=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_single_batch_loader(
    batch_size: int = 16,
    image_size: int = 28,
) -> DataLoader:
    x = torch.randn(
        batch_size,
        1,
        image_size,
        image_size,
    ).clamp(-1.0, 1.0)

    dataset = TensorDataset(x)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, x:torch.Tensor, cond:torch.Tensor):
        super().__init__()
        self.x = x
        self.cond = cond

    def __len__(self) -> int:
        return self.x.shape[0]
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "x": self.x[index],
            "cond": self.cond[index]
        }
    
def parameter_norm(model: torch.nn.Module) -> float:
    squared_sum = 0.0

    for parameter in model.parameters():
        squared_sum += parameter.detach().pow(2).sum().item()

    return squared_sum ** 0.5
    

if __name__ == '__main__':
    seed_everything()

    batch_size = 8

    x0 = torch.randn([batch_size, 1, 28, 28])

    time_steps = torch.full(
        (batch_size,),
        50,
        dtype=torch.long,
    )

    cond = torch.ones_like(time_steps)

    noise = torch.randn_like(x0)

    loader = DataLoader(
        DictDataset(x0, cond=cond),
        batch_size=batch_size,
        shuffle=False,
    )

    model = SimpleUNet()
    objective = DDPMobjective(1000, )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
  
    losses = []

    batch = next(iter(loader))

    for step in range(100):
        optimizer.zero_grad(set_to_none=True)
        output = objective.compute_loss_with_fixed_noise(model=model, batch=batch, time_steps=time_steps, noise=noise)
    
        loss = output.loss

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )
            
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        losses.append(loss.item())

        if step % 25 == 0:
            param_norm = parameter_norm(model=model)
            print(
                f"step={step:04d},"
                f"loss={loss.item():.6f},"
                f"grad_norm={float(grad_norm):.6f},"
                f"param_norm={param_norm:.6f}"
            )

    initial_loss = losses[0]
    final_loss = losses[-1]

    assert final_loss < 1e-3, (
        f"Single-batch overfitting failed: "
        f"initial_loss={initial_loss:.6e}, "
        f"final_loss={final_loss:.6e}"
    )

    assert final_loss < initial_loss * 1e-2