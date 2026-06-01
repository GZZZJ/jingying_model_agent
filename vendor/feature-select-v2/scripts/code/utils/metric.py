import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score



def ks_auc(pred_prob, true_y, pos_label=1, sample_weight=None, **kwargs):
    '''
    计算ks和auc(二分类)

    :pred_prob: numpy array, 预测概率(一维)
    :true_y: numpy array, y标签
    :pos_label: int, 正样本标签
    :sample_weight: numpy array, 样本权重
    :kwargs: 其他sklearn.metrics.roc_curve的参数(只适用于二分类)
    '''
    fpr, tpr, thresholds = roc_curve(y_true=true_y, y_score=pred_prob, pos_label=pos_label, sample_weight=sample_weight, **kwargs)
    ks_rlt = (tpr - fpr).max()
    auc_rlt = roc_auc_score(y_true=true_y, y_score=pred_prob, sample_weight=sample_weight)

    cutoff_idx = (tpr - fpr).argmax()
    cutoff = thresholds[cutoff_idx]

    return ks_rlt, auc_rlt, cutoff


def plot_roc_multi(y_true, pred_prob_dict, title='data', color_dict=None, sample_weight=None, **kwargs):
    '''
    绘制roc曲线, 多个模型绘制在一张图里

    :y_true: numpy array 或 dict, y标签, 一个样本是同一个y; dict时key与pred_prob_dict对应
    :pred_prob_dict: dict, 每个模型的预测概率, {样本集名称: pred_prob}
    :title: str, roc曲线标题
    :color_dict: dict, 每个模型的曲线的颜色, {样本集名称: 颜色}
    :sample_weight: numpy array, 样本权重, 一个样本是同一个权重
    :kwargs: 其他sklearn.metrics.roc_curve和ks_auc的其他参数
    '''
    fig = plt.figure(figsize=(8, 8))

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.plot([0, 1], [0, 1], '--')

    for k, v in pred_prob_dict.items():
        if isinstance(y_true, dict):
            y_true_sub = y_true[k]
        else:
            y_true_sub = y_true

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true_sub, v, sample_weight=sample_weight, **kwargs)
        ks, auc, cutoff = ks_auc(v, y_true_sub, sample_weight=sample_weight, **kwargs)

        if color_dict:
            plt.plot(false_positive_rate, true_positive_rate, label='DATA={}, AUC = {:.4f}, KS= {:.4f}'.format(k, auc, ks), color=color_dict[k])
        else:
            plt.plot(false_positive_rate, true_positive_rate, label='DATA={}, AUC = {:.4f}, KS= {:.4f}'.format(k, auc, ks))

    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Receiver Operating Characteristic Curve - {}'.format(title), fontsize=14)
    plt.legend(loc="lower right", fontsize=12)

    plt.show()

    return fig
