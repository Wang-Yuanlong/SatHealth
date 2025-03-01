import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, average_precision_score

metric_fn = {'auroc': roc_auc_score, 'auprc': average_precision_score, 'accuracy': accuracy_score, 
             'f1': f1_score, 'precision': precision_score, 'recall': recall_score, 'recall@k': None}
default_metrics = {'multilabelBin': ['auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall', 'recall@k'], 
                   'multiclass': ['accuracy', 'f1', 'precision', 'recall'], 
                   'Binary': ['auroc', 'auprc', 'accuracy', 'f1', 'precision', 'recall']}

class Metrics:
    def __init__(self, configuration='multilabelBin', metrics=None):
        self.configuration = configuration
        assert configuration in ['multilabelBin', 'multiclass', 'Binary']
        if metrics is not None:
            assert all([metric in metric_fn.keys() for metric in metrics]), f'Invalid metric name for {configuration} configuration'
            self.metrics = metrics
        else:
            self.metrics = default_metrics[configuration]
        
    def __call__(self, y_true, y_pred):
        self.check_input_shape(y_true, y_pred)
        if self.configuration == 'multilabelBin':
            return self.multilabelBin(y_true, y_pred)
        else:
            raise NotImplementedError(f'{self.configuration} is not implemented')

    def get_peryear_metrics(self, y_true, y_pred, dates, datefmt='%m/%d/%Y'):
        dates = np.array(dates)
        # years = dates.astype('datetime64[Y]').astype(int) + 1970
        years = pd.to_datetime(dates, format=datefmt).year
        unique_years = np.unique(years)
        metrics = {}
        for year in unique_years:
            idx = (years == year)
            y_true_ = y_true[idx]
            y_pred_ = y_pred[idx]
            metrics[str(year)] = self(y_true_, y_pred_)
        return metrics

    def multilabelBin(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.detach().cpu().numpy()
        else:
            y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()

        metrics = {}
        num_classes = y_true.shape[-1]
        for metric in self.metrics:
            if metric in ['f1', 'precision', 'recall']:
                kwargs = {'zero_division': 0}
            elif metric.endswith('@k'):
                continue
            else:
                kwargs = {}

            metric_values = []
            y_true_ = y_true
            y_pred_ = y_pred if metric in ['auroc', 'auprc'] else (y_pred > 0.5).astype(int)
            for i in range(num_classes):
                y_true_i = y_true_[:, i]
                y_pred_i = y_pred_[:, i]
                if y_true_i.sum() == 0 or y_true_i.sum() == len(y_true_i):
                    metric_values.append(0)
                    continue
                metric_values.append(metric_fn[metric](y_true_i, y_pred_i, **kwargs))
            metrics[metric] = metric_values
        avg_metric_values = {metric: np.mean(values) for metric, values in metrics.items()}
        metrics['avg'] = avg_metric_values
        if 'recall@k' in self.metrics:
            # k_values = [1, 5, 10, 20, 50]
            # for k in k_values:
            #     top_k = np.argsort(y_pred, axis=1)[:, -k:]
            #     top_k_true = y_true[np.arange(len(y_true))[:, None], top_k]
            #     num_gt = y_true.sum(axis=1)
            #     num_tp = top_k_true.sum(axis=1)
            #     mask = num_gt <= 0
            #     num_gt[mask] = 1
            #     num_tp[mask] = 1
            #     recall_k = num_tp / num_gt
            #     metrics[f'recall@{k}'] = recall_k.mean().astype(np.float64)
            metrics.update(self.get_recall_at_k(y_true, y_pred))
        return metrics
    
    def get_recall_at_k(self, y_true, y_pred, k_values=[1, 5, 10, 20, 50]):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.detach().cpu().numpy()
        else:
            y_pred = torch.sigmoid(torch.tensor(y_pred)).numpy()
        metrics = {}
        for k in k_values:
            top_k = np.argsort(y_pred, axis=1)[:, -k:]
            top_k_true = y_true[np.arange(len(y_true))[:, None], top_k]
            num_gt = y_true.sum(axis=1)
            num_tp = top_k_true.sum(axis=1)
            mask = num_gt <= 0
            num_gt[mask] = 1
            num_tp[mask] = 1
            recall_k = num_tp / num_gt
            metrics[f'recall@{k}'] = recall_k.mean().astype(np.float64)
        return metrics

    
    def check_input_shape(self, y_true, y_pred):
        if self.configuration == 'multilabelBin':
            assert y_true.shape == y_pred.shape, 'y_true and y_pred must have the same shape with multilabelBin configuration'
        elif self.configuration == 'multiclass':
            assert y_true.shape[:-1] == y_pred.shape, 'y_true and y_pred must have the same shape with multiclass configuration'
        elif self.configuration == 'Binary':
            assert y_true.shape == y_pred.shape, 'y_true and y_pred must have the same shape with Binary configuration'
            assert len(y_true.shape) == 1, 'y_true and y_pred must be 1D with Binary configuration'
        return True


if __name__ == '__main__':
    from pprint import pprint
    metric = Metrics(metrics=['recall@k'])
    generator = torch.Generator().manual_seed(42)
    y_true = torch.randint(2, (10, 5), generator=generator).float()
    y_pred = torch.rand(10, 5, generator=generator)
    values = metric(y_true, y_pred)
    pprint(values)