import numpy as np
import torch

def errors_numpy(gt, pred):

    valid_mask = gt > 0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))

    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()

    abs_diff = np.abs(pred_eval - gt_eval)

    mae = np.mean(abs_diff)
    rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))
    abs_rel = np.mean(abs_diff / gt_eval)

    log_abs_diff = np.abs(np.log10(pred_eval) - np.log10(gt_eval))

    log_mae = np.mean(log_abs_diff)
    log_rmse = np.sqrt(np.mean(np.power(log_abs_diff, 2)))

    return mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3

def errors(gt, pred):

    valid_mask = gt > 0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))

    delta1 = (threshold < 1.25).float().mean()
    delta2 = (threshold < 1.25 ** 2).float().mean()
    delta3 = (threshold < 1.25 ** 3).float().mean()

    abs_diff = np.abs(pred_eval - gt_eval)

    mae = torch.mean(abs_diff)
    rmse = torch.sqrt(torch.mean(np.power(abs_diff, 2)))
    abs_rel = torch.mean(abs_diff / gt_eval)

    log_abs_diff = torch.abs(np.log10(pred_eval) - np.log10(gt_eval))

    log_mae = torch.mean(log_abs_diff)
    log_rmse = torch.sqrt(torch.mean(np.power(log_abs_diff, 2)))

    return mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3


class Metrics(object):
    def __init__(self):
        self.results = {}
        self.eval_keys = ['mae', 'rmse', 'abs_rel', 'log_mae', 'log_rmse', 'delta1', 'delta2', 'delta3']
        for item in self.eval_keys:
            self.results[item] = []

    def update(self, gt, pred):
        # print(gt.shape, pred.shape)
        assert(gt.shape == pred.shape)
        mae, rmse, abs_rel, log_mae, log_rmse, delta1, delta2, delta3 = errors(gt, pred)
        for item in self.eval_keys:
            self.results[item].append(eval(item))

    def retrieve_avg(self):
        avg_results = {}
        for item in self.eval_keys:
            avg_results[item] = np.mean(self.results[item])
        return avg_results

    def display_avg(self):
        print("Evaluation Complete:")
        for item in self.eval_keys:
            print("{}: {:.4f}".format(item, np.mean(self.results[item])))