from torch.optim import lr_scheduler
from .build import LR_SCHEDULER_REGISTRY
from ..config import configurable


__all__ = ['MultiStepLR', 'CosineAnnealingLR', 'CyclicLR', 'ReduceLROnPlateau', 'OneCycleLR']


@LR_SCHEDULER_REGISTRY.register()
class MultiStepLR(lr_scheduler.MultiStepLR):
    @configurable
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1):
        super(MultiStepLR, self).__init__(optimizer,
                                          milestones,
                                          gamma,
                                          last_epoch,)

    @classmethod
    def from_config(cls, cfg, optim, step_per_epoch):
        return {
            'optimizer': optim,
            'milestones': cfg.SOLVER.LR_SCHEDULER.MILESTONES
        }


@LR_SCHEDULER_REGISTRY.register()
class CosineAnnealingLR(lr_scheduler.CosineAnnealingLR):
    @configurable
    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min=0,
                 last_epoch=-1):
        super(CosineAnnealingLR, self).__init__(optimizer,
                                                T_max,
                                                eta_min,
                                                last_epoch,)

    @classmethod
    def from_config(cls, cfg, optim, step_per_epoch):
        return {
            'optimizer': optim,
            'T_max': cfg.SOLVER.LR_SCHEDULER.T_MAX
        }


@LR_SCHEDULER_REGISTRY.register()
class CyclicLR(lr_scheduler.CyclicLR):
    @configurable
    def __init__(self,
                 optimizer,
                 base_lr,
                 max_lr,
                 step_size_up=2000,
                 step_size_down=None,
                 mode='triangular',
                 gamma=1.,
                 scale_fn=None,
                 scale_mode='cycle',
                 cycle_momentum=True,
                 base_momentum=0.8,
                 max_momentum=0.9,
                 last_epoch=-1):
        super(CyclicLR, self).__init__(optimizer,
                                       base_lr,
                                       max_lr,
                                       step_size_up,
                                       step_size_down,
                                       mode,
                                       gamma,
                                       scale_fn,
                                       scale_mode,
                                       cycle_momentum,
                                       base_momentum,
                                       max_momentum,
                                       last_epoch)

    @classmethod
    def from_config(cls, cfg, optim, step_per_epoch):
        return {
            'optimizer': optim,
            'base_lr': cfg.SOLVER.OPTIMIZER.WARMUP_LR,
            'max_lr': cfg.SOLVER.OPTIMIZER.MAX_LR,
        }


@LR_SCHEDULER_REGISTRY.register()
class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    @configurable
    def __init__(self,
                 optimizer,
                 mode='min',
                 actor=0.1,
                 patience=10,
                 verbose=True,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 eps=1e-8):
        super(ReduceLROnPlateau, self).__init__(optimizer,
                                                mode,
                                                actor,
                                                patience,
                                                verbose,
                                                threshold,
                                                threshold_mode,
                                                cooldown,
                                                min_lr,
                                                eps)

    @classmethod
    def from_config(cls, cfg, optim, step_per_epoch):
        return {
            'optimizer': optim,
            'mode': cfg.SOLVER.LR_SCHEDULER.MODE,
            'actor': cfg.SOLVER.LR_SCHEDULER.ACTOR,
            'patience': cfg.SOLVER.LR_SCHEDULER.PATIENCE,
        }


@LR_SCHEDULER_REGISTRY.register()
class OneCycleLR(lr_scheduler.OneCycleLR):
    @configurable
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        super(OneCycleLR, self).__init__(optimizer,
                                         max_lr,
                                         total_steps,
                                         epochs,
                                         steps_per_epoch,
                                         pct_start,
                                         anneal_strategy,
                                         cycle_momentum,
                                         base_momentum,
                                         max_momentum,
                                         div_factor,
                                         final_div_factor,
                                         last_epoch)

    @classmethod
    def from_config(cls, cfg, optim, step_per_epoch):
        return {
            'optimizer': optim,
            'max_lr': cfg.SOLVER.OPTIMIZER.LR,
            'epochs': cfg.MAX_EPOCH+1,
            'steps_per_epoch': step_per_epoch,
            'pct_start': 0.3,
            'anneal_strategy': 'cos',
        }
