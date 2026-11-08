"""Read-only per-episode physics-overflow audit, captured before automatic reset."""
import torch
from outcome_audit import audited_outcome as physical_audit


def audited_outcome(original, measure):
    base=physical_audit(original,measure)

    class NumericallyAuditedOutcome(base):
        def __init__(self,cfg,env):
            super().__init__(cfg,env)
            self.audit_overflow_mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.long)
            self.audit_after_overflow_steps=torch.zeros_like(self.audit_overflow_mask)

        def __call__(self,env):
            fresh=self.stamp!=env.common_step_counter
            result=super().__call__(env)
            # WarpBridge exposes the documented per-world overflow bitmask.
            # Read only: normal simulation reset clears the live array.
            mask=env.sim.data.overflow.to(torch.long)
            self.audit_overflow_mask[fresh]|=mask[fresh]
            self.audit_after_overflow_steps+=(fresh&(mask!=0)).long()
            return result

        def reset(self,env_ids=None):
            if hasattr(self,'audit_overflow_mask'):
                ids=slice(None) if env_ids is None else env_ids
                for name,value in dict(numerical_overflow_mask=self.audit_overflow_mask,
                    control_steps_after_numerical_overflow=self.audit_after_overflow_steps).items():
                    if name not in self.audit_last: self.audit_last[name]=torch.zeros_like(value)
                    self.audit_last[name][ids]=value[ids]
                    value[ids]=0
            return super().reset(env_ids)

    return NumericallyAuditedOutcome
