# Trainging Flow

## Entry point

scripts/train.py

## Data

Dataset -> DataLoader -> batch -> batch['x'] (shape: [B, C, H, W]), cond=batch.get("cond", None)

## Objective

batch["x"], cond -> sample timestep -> construct x_t -> model(x_t, t, cond=cond) -> calculate target -> calculate loss

## Trainer

loss -> loss scale(if scaler availabel) -> loss backward -> gradient clipping -> optimizer step -> checkpoint

## Tensor shapes

x_0 : [B, C, H, W], t:[B], x_t: [B, C, H, W], prediction: [B, C, H, W], loss: scalar.