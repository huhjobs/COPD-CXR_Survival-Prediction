import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
# from segmentation_models_pytorch.utils.train import Epoch


class mtEpoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        for loss in self.loss:
            loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
#         loss_meter = AverageValueMeter()
        loss_meters = {loss.__name__: AverageValueMeter() for loss in self.loss}
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, (y,z) in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device) #, dtype=torch.int64)
                loss, y_pred, z_pred = self.batch_update(x, y, z)
                
                # update loss logs
                loss_value = loss[0].cpu().detach().numpy()
                loss_meters[self.loss[0].__name__].add(loss_value)
                loss_value = loss[1].cpu().detach().numpy()
                loss_meters[self.loss[1].__name__].add(loss_value)
                
                loss_logs = {k: v.mean for k, v in loss_meters.items()}
                logs.update(loss_logs)

                # update metrics logs
                metric_value = self.metrics[0](y_pred, y).cpu().detach().numpy()
                metrics_meters[self.metrics[0].__name__].add(metric_value)
                metric_value = self.metrics[1](z_pred, z).cpu().detach().numpy()
                metrics_meters[self.metrics[1].__name__].add(metric_value)
                
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(mtEpoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        prediction_y = prediction[0]
        prediction_z = prediction[1]
        loss1, loss2 = self.loss[0](prediction_y, y), self.loss[1](prediction_z, z)
        loss = loss1 + loss2
        loss.backward()
        self.optimizer.step()
        return [loss1, loss2], prediction_y, prediction_z


class ValidEpoch(mtEpoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, z):
        with torch.no_grad():
            prediction = self.model.forward(x)
            prediction_y = prediction[0]
            prediction_z = prediction[1]
            loss1, loss2 = self.loss[0](prediction_y, y), self.loss[1](prediction_z, z)
            loss = loss1 + loss2
        return [loss1, loss2], prediction_y, prediction_z