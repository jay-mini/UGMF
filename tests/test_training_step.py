import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from ugm.objectives.ddpm import DDPMobjective
from ugm.samplers.ddpm import DDPMAncestralSampler
from ugm.models.unet import SimpleUNet

class ZeroVelocityModel(nn.Module):
    def forward(self, x, t, cond=None):
        return torch.zeros_like(x)
    
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
    

def test_training_step_updates_parameters(batch):
    model = SimpleUNet()
    objective = DDPMobjective()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer.zero_grad()

    output = objective.compute_loss(
        model,
        batch,
    )

    output.loss.backward()
    optimizer.step()

    changed = []

    for name, parameter in model.named_parameters():
        changed.append(
            not torch.allclose(
                before[name],
                parameter.detach(),
            )
        )

    assert any(changed)


if __name__ == '__main__':
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

    batch = next(iter(loader))

    test_training_step_updates_parameters(batch=batch)