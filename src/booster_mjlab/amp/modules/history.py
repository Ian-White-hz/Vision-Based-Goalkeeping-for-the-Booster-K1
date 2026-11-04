import torch


class TwoFrameAmpHistory:
    '''A class to maintain a history of two consecutive frames of AMP (Adaptive Motion Planning) data for each environment.'''
    def __init__(self, initial_q: torch.Tensor):
        if initial_q.ndim != 2 or initial_q.shape[1] != 22:
            raise ValueError(
                f"Expected initial_q to be a 2D tensor, got shape {initial_q.shape}"
            )
        if initial_q.dtype != torch.float32:
            raise ValueError(
                f"Expected initial_q to be of type torch.float32, got {initial_q.dtype}"
            )

        self._previous_q = initial_q.detach().clone()

    def push(
        self,
        next_q: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if next_q.shape != self._previous_q.shape:
            raise ValueError(
                f"Expected next_q to have shape {self._previous_q.shape}, got {next_q.shape}"
            )
        if (
            next_q.dtype != self._previous_q.dtype
            or next_q.device != self._previous_q.device
        ): 
            raise ValueError(
                f"Expected next_q to have the same dtype and device as previous_q, "
                f"got dtype {next_q.dtype} and device {next_q.device}, "
                f"while previous_q has dtype {self._previous_q.dtype} and device {self._previous_q.device}"
            )
        if (
            dones.shape != (next_q.shape[0],)
            or dones.dtype != torch.bool
            or dones.device != next_q.device
        ):
            raise ValueError("dones must be bool (N,) on the same device")

        next_q = next_q.detach()

        windows = torch.stack(
            [self._previous_q, next_q],
            dim=1,
        )

        # 对于 dones 为 True 的样本，使用下一帧的关节角度作为填充windows中的两个帧。
        windows[dones] = next_q[dones].unsqueeze(1)
        valid_mask = ~dones

        self._previous_q.copy_(next_q)

        return windows, valid_mask


def select_valid_amp_samples(
    windows: torch.Tensor,
    valid_mask: torch.Tensor,
    motion_ids_before: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if windows.ndim != 3 or windows.shape[1:] != (2, 22):
        raise ValueError(
            f"Expected windows to be of shape (N, 2, 22), got {windows.shape}"
        )
    if windows.dtype != torch.float32:
        raise ValueError(
            f"Expected windows to be of type torch.float32, got {windows.dtype}"
        )

    num_envs = windows.shape[0]

    if (
        valid_mask.shape != (num_envs,)
        or valid_mask.dtype != torch.bool
        or valid_mask.device != windows.device
    ):
        raise ValueError(
            f"Expected valid_mask to be bool (N,) on the same device as windows, "
            f"got shape {valid_mask.shape}, dtype {valid_mask.dtype}, device {valid_mask.device}"
        )

    if (
        motion_ids_before.shape != (num_envs,)
        or motion_ids_before.dtype != torch.int64
        or motion_ids_before.device != windows.device
    ):
        raise ValueError(
            f"Expected motion_ids_before to be int64 (N,) on the same device as windows, "
            f"got shape {motion_ids_before.shape}, dtype {motion_ids_before.dtype}, device {motion_ids_before.device}"
        )

    if ((motion_ids_before < 0) | (motion_ids_before > 5)).any():
        raise ValueError(
            f"motion_ids_before must be in the range [0, 5], got {motion_ids_before}"
        )

    features = windows[valid_mask].flatten(start_dim=1).detach()
    motion_ids = motion_ids_before[valid_mask]

    return features, motion_ids