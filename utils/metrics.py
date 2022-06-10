

from pyrsistent import v
from sqlalchemy import intersect
import numpy as np
import importlib
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay, plot_confusion_matrix
import matplotlib.pyplot as plt


def binary_dsc(target, pred):
    if np.sum(target)==0 and np.sum(pred)==0:
        return 1.0
    intersection = np.sum(target*pred)
    return (2*intersection) / (np.sum(target) + np.sum(pred))



def check_zero_division(func):
    def warp():
        
        func()
    return warp
    

def sensitivity(prediction, label):
    pass


def specificity(prediction, label):
    pass


# TODO: w/ label and w/o label
# TODO: multi-classes example
# TODO: decorate with result printing, zero-division judgeing
def precision(tp, fp):
    denominator = tp + fp
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def recall(tp, fn):
    denominator = tp + fn
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def accuracy(tp, fp, fn, tn):
    denominator = tp + fp + tn + fn
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return (tp + tn) / denominator


def f1(tp, fp, fn):
    denominator = (2 * tp + fp + fn)
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return 2 * tp / denominator


def iou(tp, fp, fn):
    denominator = (tp + fp + fn)
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tp / denominator


def specificity(tn, fp):
    denominator = tn + fp
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))
    return tn / denominator


class ClassificationMetrics():
    # TODO: macro, micro average
    # TODO: assemble to a higer API to do CV
    def __init__(self, n_class):
        self.eval_result = {}
        self.n_class = n_class

    def eval(self, y_true, y_pred):
        if self.n_class is None:
            self.n_class = np.max(np.concatenate([y_true, y_pred])) + 1
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(0, self.n_class))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    #   display_labels=n_class
                                        )
        disp.plot()
        plt.savefig('plot/cm.png')
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        tp = np.diag(cm)
        tn = tp[::-1]
        fp = np.sum(cm, axis=0) - tp
        specificity = tn / (tn + fp)

        # TODO: check specificity coorectness
        # specificity = recall_score(y_true, y_pred, pos_label=2, average=None)
        accuracy = accuracy_score(y_true, y_pred)

        print(f'Precision: {precision}')
        print(f'mean Precision: {np.mean(precision)*100:.02f} %')
        print(f'Recall: {recall}')
        print(f'mean Recall: {np.mean(recall)*100:.02f} %')
        print(f'Specificity: {specificity}')
        print(f'mean Specificity: {np.mean(specificity)*100:.02f} %')
        print('Accuracy', accuracy)
        print(cm)


def cls_metrics(y_true, y_pred, save_name='cm.png', n_class=None):
    if n_class is None:
        n_class = np.max(np.concatenate([y_true, y_pred])) + 1

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(0, n_class))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                #   display_labels=n_class
                                  )
    disp.plot()
    plt.savefig(save_name)

    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    tp = np.diag(cm)
    tn = np.repeat(np.sum(tp), tp.size) - tp
    fp = np.sum(cm, axis=0) - tp
    specificity = tn / (tn + fp)
    # TODO: change it because true_divide will trigger error
    specificity = np.where(np.isnan(specificity), 0.0, specificity)
    accuracy = accuracy_score(y_true, y_pred)

    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_specificity = np.mean(specificity)

    print(20*'=')
    print(f'Precision: {precision}')
    print(f'mean Precision: {mean_precision*100:.02f} %')
    print(f'Recall: {recall}')
    print(f'mean Recall: {mean_recall*100:.02f} %')
    print(f'Specificity: {specificity}')
    print(f'mean Specificity: {mean_specificity*100:.02f} %')
    print(f'Accuracy {accuracy}')
    print(f'Confuion matrix {cm}')
    return mean_precision, mean_recall, mean_specificity, accuracy, cm


# TODO: property for all avaiable metrics
# TODO: should implement in @staticmethod

# TODO: property for all avaiable metrics
# TODO: should implement in @staticmethod
class SegmentationMetrics():
    def __init__(self, num_class, metrics=None):
        # TODO: parameter check
        self.num_class = num_class
        self.total_tp = None
        self.total_fp = None
        self.total_fn = None
        self.total_tn = None
        # TODO: check value and class (Dose sklearn func do this part?)
        # TODO: check shape
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = ['precision', 'recall', 'specificity', 'accuracy', 'f1', 'iou']
        
    def __call__(self, label, pred):
        self.label = label
        self.pred = pred
        self.tp, self.fp, self.fn, self.tn = self.cm_value()
        self.total_tp = self.tp if self.total_tp is None else self.total_tp + self.tp
        self.total_fp = self.fp if self.total_fp is None else self.total_fp + self.fp
        self.total_fn = self.fn if self.total_fn is None else self.total_fn + self.fn
        self.total_tn = self.tn if self.total_tn is None else self.total_tn + self.tn

        eval_result = {}
        for m in self.metrics:
            if m == 'precision':
                eval_result[m] = precision(self.tp, self.fp)
            elif m == 'recall':
                eval_result[m] = recall(self.tp, self.fn)
            elif m == 'specificity':
                eval_result[m] = specificity(self.tn, self.fp)
            elif m == 'accuracy':
                eval_result[m] = accuracy(self.tp, self.fp, self.fn, self.tn)
            elif m == 'f1':
                eval_result[m] = f1(self.tp, self.fp, self.fn)
            elif m == 'iou':
                eval_result[m] = iou(self.tp, self.fp, self.fn)
        return eval_result

    def confusion_matrix(self):
        num_class = self.num_class if self.num_class > 1 else self.num_class + 1
        self.label = np.reshape(self.label, [-1])
        self.pred = np.reshape(self.pred, [-1])
        cm = confusion_matrix(self.label, self.pred, labels=np.arange(0, num_class))
        return cm
    
    # def cm_value(self):
    #     cm = self.confusion_matrix()
    #     tp = cm[1,1]
    #     fp = cm[0,1]
    #     fn = cm[1,0]
    #     tn = cm[0,0]
    #     return (tp, fp, fn, tn)

    def cm_value(self):
        cm = self.confusion_matrix()
        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp
        tn = np.array([np.sum(tp)]*(tp.size)) - tp
        return (tp, fp, fn, tn)


# def torch_confusion_matrix
def get_evaluation_metric(config):
    """
    Returns the evaluation metric function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'eval_metric' key
    :return: an instance of the evaluation metric
    """

    def _metric_class(class_name):
        m = importlib.import_module('utils.metrics')
        clazz = getattr(m, class_name)
        return clazz

    assert 'eval_metric' in config, 'Could not find evaluation metric configuration'
    metric_config = config['eval_metric']
    metric_class = _metric_class(metric_config['name'])
    return metric_class(**metric_config)


# TODO: get Confusion matrix
# TODO: flexible for single sample test
# TODO: flexibility for tensorflow
class BaseEvaluator():
    def __init__(self, loader, net, metrics_name, data_keys=['input', 'gt'], *args, **kwargs):
        self.loader = loader
        self.net = net
        self.metrics_name = metrics_name
        self.data_keys = data_keys

    def get_evaluation(self):
        input_key, gt_key = self.data_keys
        metrics_func = self.get_metrics()
        for _, data in enumerate(self.loader):
            inputs, labels = data[input_key], data[gt_key]
            outputs = self.net(inputs)
            metrics_func(labels, outputs)
            evaluation = self.aggregation()
        return evaluation

    def get_metrics(self):
        def metrics(label, pred):
            self.total_cm = 0
            for m in self.metrics_name:
                if m in ('precsion', 'recall'):
                    cm = confusion_matrix(label, pred)
                else:
                    raise ValueError('Undefined metrics name.')
                self.total_cm += cm

        return metrics

    def aggregation(self):
        pass 

    def check_format(self):
        pass

    def check_shape(self):
        pass


