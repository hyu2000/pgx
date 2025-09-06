from pydantic import BaseModel
import pgx


class Config(BaseModel):
    env_id: pgx.EnvId = "go_5x5C2"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    num_channels: int = 64  # 128
    num_layers: int = 6
    resnet_v2: bool = True
    bn_momentum: float = 0.9
    bn_mode: str = 'batch'
    # selfplay params
    selfplay_batch_size: int = 512   # #games
    num_simulations: int = 32
    max_num_steps: int = 64  # max_num_moves per game
    # training params
    training_batch_size: int = 256   # #samples per mini-batch
    learning_rate: float = 0.01
    lr_decay_steps: int = 256  # halve LR every #steps
    weight_decay: float = 1e-4
    # eval params
    eval_interval: int = 5
    eval_batch_size: int = 128  # #games
    baseline: str = 'go_5x5C2_250906-125418/000075.ckpt'
    checkpoint_interval: int = 5
    wandb_project: str = "pgx-az"

    class Config:
        extra = "forbid"
