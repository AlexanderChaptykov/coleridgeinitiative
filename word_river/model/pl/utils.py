import numbers
import os

import torch
import wandb
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.utilities import rank_zero_only

from word_river.cli_parser.train import training_args, data_args, model_args, wandb_args


class BestMetrics(Callback):
    """
    Tracks best metrics - at the end of training logs best epoch metrics
    Required for a proper sweep
    """

    def __init__(self, monitor="f1_score", mode="max"):
        super().__init__()
        self.best_metrics = {}
        self.monitor = monitor
        self.mode = mode

    def on_fit_end(self, trainer, pl_module):
        wandb.log(self.best_metrics)

    def on_epoch_end(self, trainer, pl_module):
        current_metric_value = trainer.callback_metrics[self.monitor]
        if self.mode == 'min':
            best_metric_value = self.best_metrics.get(f'best_{self.monitor}', float('inf'))
            # new metric score is higher than best value (minimization goal)
            if current_metric_value < best_metric_value:
                self.best_metrics = {f'best_{k}': v for k, v in trainer.callback_metrics.items()}
        else:
            best_metric_value = self.best_metrics.get(f'best_{self.monitor}', float('-inf'))
            # new metric score is higher than best value (maximization goal)
            if current_metric_value > best_metric_value:
                self.best_metrics = {f'best_{k}': v for k, v in trainer.callback_metrics.items()}


class Finetune(EarlyStopping):
    """
    Allows model finetuning:
    1. Train only sent_embeds and sent_predictor.
    2. Train only word_embeds
    3. Train whole model
    Next stage is decided in early stopping fashion
    """

    def __init__(self, *args, **kwargs):
        super(Finetune, self).__init__(*args, **kwargs)
        self.stage = 0
        self.best_model_path = None

    def on_save_checkpoint(self, trainer, pl_module):
        self.best_model_path = os.path.join(training_args.output_dir, wandb.run.name,
                                            f'cancellation-binary-model-epoch={trainer.current_epoch}.ckpt')

    def on_fit_start(self, trainer, pl_module):
        """
        Run on fit start
        """
        self.finetune(trainer, pl_module)

    def _run_early_stopping_check(self, trainer, pl_module):
        """
        Checks whether the finetune condition is met
        and if so tells the model to start next finetuning stage
        """
        logs = trainer.callback_metrics

        if (
                trainer.fast_dev_run  # disable early_stopping with fast_dev_run
                or not self._validate_condition_metric(logs)  # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        current = logs.get(self.monitor)

        # when in dev debugging
        trainer.dev_debugger.track_early_stopping_history(self, current)

        if current is not None:
            if isinstance(current, Metric):
                current = current.compute()
            elif isinstance(current, numbers.Number):
                current = torch.tensor(current, device=pl_module.device, dtype=torch.float)

        if trainer.use_tpu and TPU_AVAILABLE:
            current = current.cpu()

        if self.monitor_op(current - self.min_delta, self.best_score):
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            should_start_next_stage = self.wait_count >= self.patience

            if bool(should_start_next_stage):
                # start counting again
                self.wait_count = 0
                self.best_score = current
                # moving to the next finetuning stage
                self.stage += 1
                if self.stage <= 2:
                    self.finetune(trainer, pl_module)
                else:
                    self.stopped_epoch = trainer.current_epoch
                    trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.accelerator_backend.early_stopping_should_stop(pl_module)
        trainer.should_stop = should_stop

    def finetune(self, trainer, pl_module):
        # load best weights
        if self.best_model_path is not None and os.path.exists(self.best_model_path) and training_args.save_checkpoints:
            pl_module.load_state_dict(torch.load(self.best_model_path), strict=False)
        # reset schedulers
        for scheduler in trainer.lr_schedulers:
            scheduler['scheduler']._reset()
        self.freeze_model(pl_module)

    def freeze_model(self, pl_module):
        if self.stage == 0:
            # freeze word_embeds
            for p in pl_module.model.word_embeds.parameters():
                p.requires_grad = False
        elif self.stage == 1:
            # unfreeze word_embeds
            for p in pl_module.model.word_embeds.parameters():
                p.requires_grad = True
            # freeze sent_embeds and sent_predictor
            for p in pl_module.model.sent_embeds.parameters():
                p.requires_grad = False
            for p in pl_module.model.sent_predictor.parameters():
                p.requires_grad = False
        else:
            # unfreeze everything
            for p in pl_module.model.sent_embeds.parameters():
                p.requires_grad = True
            for p in pl_module.model.sent_predictor.parameters():
                p.requires_grad = True


class PrintLogger(DummyLogger):
    """ Dummy logger for internal use. Is usefull if we want to disable users
        logger for a feature, but still secure that users code can run """
    def __init__(self):
        self.all_m = []

    @rank_zero_only
    def log_metrics(self, metrics, step):
        print(metrics)
        self.all_m.append(metrics)
        print(step)
        print()



@rank_zero_only
def get_callbacks():
    callbacks = []
    if training_args.finetune:
        finetune_callback = Finetune(
            monitor=training_args.callbacks_monitor, min_delta=0.0,
            patience=training_args.finetune_patience,
            verbose=True,
            mode=training_args.callbacks_mode)
        callbacks.append(finetune_callback)
    else:
        early_stop_callback = EarlyStopping(
            monitor=training_args.callbacks_monitor, min_delta=0.0,
            patience=training_args.early_stopping_patience,
            verbose=True,
            mode=training_args.callbacks_mode
        )
        callbacks.append(early_stop_callback)

    best_metrics_callback = BestMetrics(monitor=training_args.callbacks_monitor, mode=training_args.callbacks_mode)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.extend([lr_monitor, best_metrics_callback])

    if training_args.save_checkpoints:
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(training_args.output_dir, wandb.run.name,
                                  'cancellation-binary-model-{epoch}'),
            save_top_k=training_args.save_top_k,
            monitor=training_args.callbacks_monitor,
            mode=training_args.callbacks_mode,
            verbose=True
        )
        callbacks.append(checkpoint_callback)
    return callbacks


@rank_zero_only
def get_loggers():
    wandb_logger = WandbLogger(
        project=wandb_args.project,
        experiment=wandb.init(
            project=wandb_args.project,
            entity=wandb_args.entity
        ),
        config={
            **model_args.__dict__, **data_args.__dict__, **training_args.__dict__
        }
    )
    return [wandb_logger]
