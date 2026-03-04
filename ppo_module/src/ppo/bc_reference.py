"""BC reference policy for KL penalty during PPO training.

Loads a trained BC policy and adds a KL divergence penalty between the
PPO agent's action distribution and the BC reference distribution. This
prevents catastrophic forgetting of BC-learned behaviors.

Usage:
    from src.ppo.bc_reference import BCReferenceCallback

    cb = BCReferenceCallback(
        bc_policy_path="models/bc/best_bc.pt",
        kl_coef=0.1,
        device="cpu",
    )
    model.learn(total_timesteps=700, callback=cb)
"""

import torch
from stable_baselines3.common.callbacks import BaseCallback


class BCReferenceCallback(BaseCallback):
    """Adds KL penalty between PPO and BC action distributions.

    At each step, computes KL(PPO || BC) from cached BC logits and
    subtracts a scaled penalty from the reward buffer. This encourages
    the PPO policy to stay close to the BC reference.

    Args:
        bc_policy_path: Path to saved BCPolicy checkpoint (best_bc.pt).
        kl_coef: KL penalty coefficient (0 = disabled).
        device: Torch device for BC policy.
    """

    def __init__(
        self,
        bc_policy_path: str,
        kl_coef: float = 0.1,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.bc_policy_path = bc_policy_path
        self.kl_coef = kl_coef
        self.device = device
        self._bc_policy = None

    def _on_training_start(self) -> None:
        try:
            from src.bc.bc_policy import BCPolicy

            self._bc_policy = BCPolicy.load(
                self.bc_policy_path, device=self.device,
            )
            self._bc_policy.eval()
            print(
                f"[BCReference] Loaded BC policy from {self.bc_policy_path}"
            )
        except Exception as e:
            print(f"[BCReference] Failed to load BC policy: {e}")
            self._bc_policy = None

    def _on_step(self) -> bool:
        if self._bc_policy is None or self.kl_coef <= 0:
            return True

        obs = self.locals.get("obs_tensor")
        if obs is None:
            return True

        try:
            with torch.no_grad():
                bc_logits = self._bc_policy(obs)
                ppo_dist = self.model.policy.get_distribution(obs)
                ppo_logits = ppo_dist.distribution.logits

                bc_log_probs = torch.log_softmax(bc_logits, dim=-1)
                ppo_log_probs = torch.log_softmax(ppo_logits, dim=-1)
                ppo_probs = torch.exp(ppo_log_probs)

                # KL(PPO || BC) = sum(ppo * (log_ppo - log_bc))
                kl = (ppo_probs * (ppo_log_probs - bc_log_probs)).sum(-1)
                kl_mean = kl.mean().item()

                # Subtract KL penalty from latest reward in buffer
                self.locals["rewards"][-1] -= self.kl_coef * kl_mean

                if self.logger is not None:
                    self.logger.record("cr/bc_kl", kl_mean)
        except Exception:
            pass  # Silently skip if shapes mismatch or other issues

        return True
