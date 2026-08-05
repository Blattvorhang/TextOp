import os
import glob


def _is_rank_zero():
    """Check if this is the main DDP process (or running without DDP)."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return local_rank == 0


class TrainPlatform:
    def __init__(self, save_dir, *args, **kwargs):
        self.path, file = os.path.split(save_dir)
        self.name = kwargs.get('name', file)

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_media(self, title, series, iteration, local_path):
        pass

    def report_figure(self, name, figure, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        if _is_rank_zero():
            from clearml import Task
            path, name = os.path.split(save_dir)
            self.task = Task.init(project_name='RL',
                                  task_name=name)
            self.logger = self.task.get_logger()
        else:
            self.task = None
            self.logger = None

    def report_scalar(self, name, value, iteration, group_name):
        if self.logger is not None:
            self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_media(self, title, series, iteration, local_path):
        if self.logger is not None:
            self.logger.report_media(title=title, series=series, iteration=iteration, local_path=local_path)

    def report_args(self, args, name):
        if self.task is not None:
            self.task.connect(args, name=name)

    def close(self):
        if self.task is not None:
            self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        if _is_rank_zero():
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=save_dir)
        else:
            self.writer = None

    def report_scalar(self, name, value, iteration, group_name=None):
        if self.writer is not None:
            self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def report_figure(self, name, figure, iteration, group_name=None):
        tag = f'{group_name}/{name}' if group_name else name
        self.writer.add_figure(tag, figure, global_step=iteration, close=True)

    def close(self):
        if self.writer is not None:
            self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir, *args, **kwargs):
        pass


class WandBPlatform(TrainPlatform):
    import wandb
    def __init__(self, save_dir, config=None, *args, **kwargs):
        super().__init__(save_dir, *args, **kwargs)
        if _is_rank_zero():
            self.wandb.login(host=os.getenv("WANDB_BASE_URL"), key=os.getenv("WANDB_API_KEY"))
            self.wandb.init(
                project='RL',
                name=self.name,
                id=self.name,
                resume='allow',
                entity='tau-motion',
                save_code=True,
                config=config)
            self._wandb_initialized = True
        else:
            self._wandb_initialized = False

    def report_scalar(self, name, value, iteration, group_name=None):
        if self._wandb_initialized:
            self.wandb.log({name: value}, step=iteration)

    def report_media(self, title, series, iteration, local_path):
        if self._wandb_initialized:
            files = glob.glob(f'{local_path}/*.mp4')
            self.wandb.log({series: [self.wandb.Video(file, format='mp4', fps=20) for file in files]}, step=iteration)

    def report_args(self, args, name):
        if self._wandb_initialized:
            self.wandb.config.update(args)

    def watch_model(self, *args, **kwargs):
        if self._wandb_initialized:
            self.wandb.watch(args, kwargs)

    def close(self):
        if self._wandb_initialized:
            self.wandb.finish()

