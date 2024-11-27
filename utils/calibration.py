import numpy as np
import torch

from transcal.utils import get_weight, ECELoss, VectorOrMatrixScaling, TempScaling, CPCS, TransCal, Oracle
from transcal.TransCal import cal_acc_error


class ECE:

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.n_samples = 0
        self.correct = [[] for _ in range(n_bins)]
        self.conf = [[] for _ in range(n_bins)]
        self.n = [0 for _ in range(n_bins)]
    
    def update(self, probs, labels):
        self.n_samples += labels.size(0)
        confidence, preds = probs.max(dim=1)
        for conf, y_pred, y_true in zip(confidence, preds, labels):
            bin = min(int(conf * self.n_bins), self.n_bins - 1)
            self.correct[bin].append(y_pred == y_true)
            self.conf[bin].append(conf)
            self.n[bin] += 1

    def score(self, reduction="sum"):
        return np.nansum([np.abs(np.mean(self.correct[bin]) - np.mean(self.conf[bin])) * self.n[bin] / self.n_samples for bin in range(self.n_bins)])


class StaticECE:

    def __init__(self, n_classes, n_bins=10):
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.n_samples = 0
        self.correct = [[[] for _ in range(n_bins) ] for _ in range(n_classes)]
        self.conf = [[[] for _ in range(n_bins)] for _ in range(n_classes)]
        self.n = [[0 for _ in range(n_bins)] for _ in range(n_classes)]
    
    def update(self, probs, labels):
        self.n_samples += labels.size(0)
        confidence, preds = probs.max(dim=1)
        for conf, y_pred, y_true in zip(confidence, preds, labels):
            bin = min(int(conf * self.n_bins), self.n_bins - 1)
            self.correct[y_true][bin].append(y_pred == y_true)
            self.conf[y_true][bin].append(conf)
            self.n[y_true][bin] += 1

    def score(self, reduction="none"):
        cjce = np.array([
            np.nansum([np.abs(np.mean(self.correct[c][bin]) - np.mean(self.conf[c][bin])) * self.n[c][bin] / self.n_samples for bin in range(self.n_bins)])
            for c in range(self.n_classes)])
        if reduction == "none":  # Class-j-ECE
            return cjce
        elif reduction == "mean":  # Static Calibration Error
            return np.mean(cjce)


class ClasswiseECE:

    def __init__(self, n_classes, n_bins=10):
        self.n_bins = n_bins
        self.n_classes = n_classes
        self.n_samples = 0
        self.correct = [[[] for _ in range(n_bins) ] for _ in range(n_classes)]
        self.conf = [[[] for _ in range(n_bins)] for _ in range(n_classes)]
        self.n = [[0 for _ in range(n_bins)] for _ in range(n_classes)]
    
    def update(self, probs, labels):
        self.n_samples += labels.size(0)
        for prob, y_true in zip(probs, labels):
            for c in range(self.n_classes):
                bin = min(int(prob[c] * self.n_bins), self.n_bins - 1)
                self.correct[c][bin].append(y_true == c)
                self.conf[c][bin].append(prob[c])
                self.n[c][bin] += 1

    def score(self, reduction="none"):
        cjce = np.array([
            np.nansum([np.abs(np.mean(self.correct[c][bin]) - np.mean(self.conf[c][bin])) * self.n[c][bin] / self.n_samples for bin in range(self.n_bins)]) 
            for c in range(self.n_classes)])
        if reduction == "none":  # Class-j-ECE
            return cjce
        elif reduction == "mean":  # Classwise Calibration Error
            return np.mean(cjce)


def get_optimal_temp(cal_method, dataloaders, model, classifier_layer, output_name, previous_temp=None, alpha=0.0):
    """
    TODO: write docstring.
    """
    # 1. Gather features and logits
    def gather_outputs(selected_loader, output_name):
        with torch.no_grad():
            start_test = True
            for inputs, labels in selected_loader:
                inputs = inputs.cuda()
                fc_features = model(inputs)
                logit = classifier_layer(fc_features)

                if start_test:
                    features_ = fc_features.float().cpu()
                    outputs_ = logit.float().cpu()
                    labels_ = labels
                    start_test = False
                else:
                    features_ = torch.cat((features_, fc_features.float().cpu()), 0)
                    outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                    labels_ = torch.cat((labels_, labels), 0)
            return features_, outputs_, labels_

    features_source_train, logits_source_train, labels_source_train = \
        gather_outputs(dataloaders["source_train"], output_name)
    features_source_val, logits_source_val, labels_source_val = \
        gather_outputs(dataloaders["source_val"], output_name)
    features_target_train, logits_target_train, labels_target_train = \
        gather_outputs(dataloaders["target_train"], output_name)
    features_target_val, logits_target_val, labels_target_val = \
        gather_outputs(dataloaders["target_val"], output_name)

    ece_criterion = ECELoss()
    ece = {
        "ece_source_train": ece_criterion(logits_source_train, labels_source_train).item(),
        "ece_source_val": ece_criterion(logits_source_val, labels_source_val).item(),
        "ece_target_train": ece_criterion(logits_target_train, labels_target_train).item(),
        "ece_target_val": ece_criterion(logits_target_val, labels_target_val).item(),
    }

    if cal_method is None:
        return None, None, ece

    # 2. Calibrate
    if cal_method == 'VectorScaling' or cal_method == 'MatrixScaling':
        _, _, (W, b) = VectorOrMatrixScaling(logits_source_val, labels_source_val, logits_target_train, labels_target_train, cal_method=cal_method)
        optimal_temp = None
        if cal_method == 'VectorScaling':
            calibration_func = lambda logits: logits.cuda() * W + b
        elif cal_method == "MatrixScaling":
            calibration_func = lambda logits: torch.matmul(logits.cuda(), W) + b
    else:
        if cal_method == 'TempScaling':
            cal_model = TempScaling()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val)
        elif cal_method == 'CPCS':
            weight = get_weight(features_source_train, features_target_train, features_source_val)
            cal_model = CPCS()
            optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val, torch.from_numpy(weight))
        elif cal_method == 'TransCal':
            """calibrate the source model first and then attain the optimal temperature for the source dataset"""
            weight = get_weight(features_source_train, features_target_train, features_source_val)
            cal_model = TempScaling()
            optimal_temp_source = cal_model.find_best_T(logits_source_val, labels_source_val)
            _, source_confidence, error_source_val = cal_acc_error(logits_source_val / optimal_temp_source, labels_source_val)

            cal_model = TransCal(bias_term=True, variance_term=True)
            optimal_temp = cal_model.find_best_T(logits_target_train.numpy(), weight, error_source_val, source_confidence.item())
        elif cal_method == 'Oracle':
            cal_model = Oracle()
            optimal_temp = cal_model.find_best_T(logits_target_train, labels_target_train)
        if previous_temp is not None:
            optimal_temp = alpha * previous_temp + (1 - alpha) * optimal_temp
        ### CAP max temp to 100
        # optimal_temp = min(optimal_temp, 100)
        calibration_func = lambda logits: logits / optimal_temp

    ece["ece_target_train_cal"] = ece_criterion(calibration_func(logits_target_train).cpu(), labels_target_train).item()
    ece["ece_target_val_cal"] = ece_criterion(calibration_func(logits_target_val).cpu(), labels_target_val).item()

    return calibration_func, optimal_temp, ece
