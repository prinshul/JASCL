import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, first_novel_class):
        super().__init__()
        self.n_classes = n_classes
        self.first_novel_class = first_novel_class
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.class_wise_dice = np.zeros(n_classes).tolist()
        
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.cpu().detach().numpy().flatten(), lp.cpu().detach().numpy().flatten())
        self.total_samples += len(label_trues)
    
    def calculate_dice_each_class(self, class_dice):
        res_array = np.zeros(self.n_classes).tolist()
        # print("SELF:", len(self.class_wise_dice), len(class_dice))
        for i in range(len(self.class_wise_dice)):
            if class_dice[i] != 'X':
                res_array[i] = class_dice[i] + self.class_wise_dice[i]        
            else:
                res_array[i] = 'X'
        # res_array = np.array(class_dice) + np.array(self.class_wise_dice)
        # self.class_wise_dice = res_array.tolist()
        self.class_wise_dice = res_array
        return self.class_wise_dice


    def to_str_verbose(self, results):
        string = "\n"
        ignore = ["Class IoU", "Class Dice Score", "Class Acc", "Class Prec",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in results.items():
            if k not in ignore:
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Dice Score:\n'
        for k, v in results['Class Dice Score'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Prec:\n'
        for k, v in results['Class Prec'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def to_str(self, results):
        string = "\n"
        ignore = ["Class IoU", "Class Acc", "Class Prec", "Agg",
                  "Confusion Matrix Pred", "Confusion Matrix", "Confusion Matrix Text"]
        for k, v in results.items():
            if k not in ignore:
                print("k",k)
                print("v",v)
                string += "%s: %s\n" % (k, str(v))

        i = 0
        for name in ('Class IoU', 'Class Acc', 'Class Prec'):
            string += f'{name}:\n'
            string += f"\tB: {results['Agg'][i]}\n"
            string += f"\tN: {results['Agg'][i+1]}\n"
            i += 2

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        first_novel_class = self.first_novel_class

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        precision_cls_c = diag / (hist.sum(axis=0) + EPS)
        precision_cls = np.mean(precision_cls_c)
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        dice_score =  (2*diag) / (gt_sum + hist.sum(axis=0) + EPS)
        mean_iu = np.mean(iu[mask])
        mean_dice = np.mean(dice_score[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_dice = dict(zip(range(self.n_classes), [dice_score[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))
        cls_prec = dict(zip(range(self.n_classes), [precision_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        short_metrics = []
        for metric in (cls_iu, cls_dice, cls_acc, cls_prec):
            base_classes = 0.
            novel_classes = 0.
            for k, v in metric.items():
                if v != "X":
                    if k < first_novel_class:
                        base_classes += v
                    else:
                        novel_classes += v

            base = base_classes / first_novel_class
            if (self.n_classes - first_novel_class) != 0:
                novel = novel_classes / (self.n_classes - first_novel_class)
            else:
                novel = 0
            short_metrics += [base, novel]

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "Mean Precision": precision_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Mean Dice Score": mean_dice,
            "Class IoU": cls_iu,
            "Class Dice Score": cls_dice,
            "Class Acc": cls_acc,
            "Class Prec": cls_prec,
            "Agg": short_metrics
        }

    def get_conf_matrixes(self, mode):
        return {"Confusion Matrix Text": self.confusion_matrix_to_text(),
                "Confusion Matrix": self.confusion_matrix_to_fig(mode),
                "Confusion Matrix Pred": self.confusion_matrix_to_fig(mode, norm_gt=False)}

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0
        self.class_wise_dice = np.zeros(self.n_classes)

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()

    def confusion_matrix_to_fig(self, mode, norm_gt=True):
        if norm_gt:
            div = (self.confusion_matrix.sum(axis=1) + 0.000001)[:, np.newaxis]
        else:
            div = (self.confusion_matrix.sum(axis=0) + 0.000001)[np.newaxis, :]
        cm = self.confusion_matrix.astype('float') / div
       
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        if mode == 'train':
            if norm_gt:
                plt.savefig('cil/Few-Shot-DCIL/save_fig/Confusion_Matrix_Train.png')
            else:
                plt.savefig('cil/Few-Shot-DCIL/save_fig/Confusion_Matrix_Pred_Train.png')
        else:
            if norm_gt:
                plt.savefig('cil/Few-Shot-DCIL/save_fig/Confusion_Matrix_Test.png')
            else:
                plt.savefig('cil/Few-Shot-DCIL/save_fig/Confusion_Matrix_Pred_Test.png')

        fig.tight_layout()
        return fig

    def confusion_matrix_to_text(self):
        string = []
        for i in range(self.n_classes):
            string.append(f"{i} : {self.confusion_matrix[i].tolist()}")
        return "\n" + "\n".join(string)


class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]