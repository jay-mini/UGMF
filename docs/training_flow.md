# Training Flow

## Entry point

scripts/train.py

## Data

Dataset -> DataLoader -> batch -> batch['x'] (shape: [B, C, H, W]), cond=batch.get("cond", None)

For class-conditional CFG training, the trainer independently replaces each
label with `-1` with probability `cond_drop_prob`. `-1` is the per-example null
condition and produces the same model branch as `cond=None`. Condition dropout
is disabled when `class_cond: false`.

## Objective

batch["x"], cond -> sample timestep -> construct x_t -> model(x_t, t, cond=cond) -> calculate target -> calculate loss

## Trainer

For each successful optimizer update:

```text
loss
-> loss scaling (when AMP GradScaler is enabled)
-> backward
-> gradient clipping
-> optimizer.step()
-> global_step += 1
-> EMA update
-> scheduler.step()
-> optional checkpoint
```

The scheduler is stepped once per successful optimizer update. If AMP skips an
optimizer update because of an overflow, `global_step`, EMA, and the scheduler
also do not advance.

## AMP validation

CUDA `mixed_precision: fp16` enables both `torch.amp.autocast` and
`torch.amp.GradScaler`. The trainer checks that the loss is finite before
backpropagation. After a scaled update, it compares the scaler value before
and after `scaler.update()`; an overflow-induced scale reduction means the
optimizer update was skipped, so the global step, EMA, and scheduler remain
unchanged.

`tests/test_training_step.py::test_cuda_amp_has_finite_loss_and_updates_parameters`
uses a real CUDA convolution to verify that the loss is finite, the scaler is
enabled, the global step advances, and at least one parameter changes. The test
is skipped on machines without CUDA rather than pretending that CPU execution
validated fp16 CUDA AMP.

## Scheduler and resume

The training entry point supports `scheduler: none` and `scheduler: cosine`.
For cosine scheduling, `scheduler_total_steps: null` is resolved to
`num_epochs * len(train_loader)` before `config.json` is written. `max_steps`
is treated as a temporary stopping boundary and does not shorten the intended
learning-rate horizon.

Checkpoints persist model, objective, optimizer, scheduler, AMP scaler, EMA,
RNG, epoch, and global step state. A resumed checkpoint whose `global_step` is
already greater than or equal to `max_steps` is a no-op.

## Mid-epoch resume limitation

The current checkpoint format does not save the DataLoader iterator or sampler
cursor. A checkpoint saved in the middle of epoch `e` restores all stateful
training components, but restarts epoch `e` from its first batch. Consequently:

- optimizer-step, learning-rate, and EMA counters remain continuous;
- batches already consumed in that epoch may be seen again;
- exact uninterrupted sample order is guaranteed only at epoch-boundary
  checkpoints.

Mid-epoch checkpoints record `batches_consumed_in_epoch`, `resume_behavior`,
and `resume_limitation` in `checkpoint["extra"]`; loading one emits a
`RuntimeWarning`.

## Tensor shapes

x_0 : [B, C, H, W], t:[B], x_t: [B, C, H, W], prediction: [B, C, H, W], loss: scalar.
