import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from IPython.display import display, Markdown
from numpy.random import default_rng
from collections import OrderedDict
from copy import deepcopy
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer

sns.set_theme(style='whitegrid')


# utility functions
def compute_dataset_fairness_metrics(data: BinaryLabelDataset, unpriv_group: list, priv_group: list, disp=True):
    """ Computes: Disparate Impact and Statistical Parity """

    b = BinaryLabelDatasetMetric(data, unprivileged_groups=unpriv_group, privileged_groups=priv_group)
    metrics = dict()
    metrics['Disparate Impact'] = b.disparate_impact()
    metrics['Statistical Parity'] = b.statistical_parity_difference()
    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))
    return metrics


def compute_fairness_metrics(dataset_true, dataset_pred,
                             unprivileged_groups, privileged_groups,
                             disp=True):
    """ Computes: Balanced Accuracy 
    Statistical Parity Difference
    Disparate Impact
    Average odds difference
    Equal Opportunity Difference
    Theil Index """

    classified_metric_pred = ClassificationMetric(dataset_true,
                                                  dataset_pred,
                                                  unprivileged_groups=unprivileged_groups,
                                                  privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * (classified_metric_pred.true_positive_rate() +
                                          classified_metric_pred.true_negative_rate())
    metrics["Statistical parity"] = classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds"] = classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity"] = classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def compute_quality_metrics(dataset_true, dataset_pred, unprivileged_group, privileged_group):
    clm = ClassificationMetric(dataset_true, dataset_pred, unprivileged_group, privileged_group)
    p_metrics = OrderedDict()
    u_metrics = OrderedDict()
    p_metrics['Precision'] = clm.precision(privileged=True)
    p_metrics['Recall'] = clm.recall(privileged=True)
    p_metrics['F1 Score'] = (2 * p_metrics['Precision'] * p_metrics['Recall']) / (
            p_metrics['Precision'] + p_metrics['Recall'])

    u_metrics['Precision'] = clm.precision(privileged=False)
    u_metrics['Recall'] = clm.recall(privileged=False)
    u_metrics['F1 Score'] = (2 * u_metrics['Precision'] * u_metrics['Recall']) / (
            u_metrics['Precision'] + u_metrics['Recall'])
    return p_metrics, u_metrics


def build_dataset(n_samples, n_features, n_informative, n_sensitive):
    """"Builds a syntetic dataset for classification"""
    x, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative)
    data = pd.DataFrame(np.column_stack((x, y)), columns=[i for i in range(11)])
    s = np.arange(n_sensitive)
    s = np.repeat(s, n_samples / 2)
    rnd = default_rng()
    rnd.shuffle(s)
    data['s'] = s
    return data


def split_data(data: BinaryLabelDataset):
    # data = BinaryLabelDataset(df=data, label_names=label_names, protected_attribute_names=protect_attr_names)
    data_train, data_tv = data.split([0.7], shuffle=True)
    data_test, data_valid = data_tv.split([0.5], shuffle=True)
    return data_train, data_test, data_valid


def x_y_split(train, test, sensitive_attributes=[]):
    x_train = train.features
    x_test = test.features
    y_train = train.labels.ravel()
    y_test = test.labels.ravel()
    for s in sensitive_attributes:
        x_train = np.delete(train.features, train.feature_names.index(s), axis=1)
        x_test = np.delete(test.features, test.feature_names.index(s), axis=1)
    return x_train, y_train, x_test, y_test


def merge_datasets(datasets: dict):
    keys = list(datasets.keys())
    first_data = datasets.pop(keys[0])
    merged_metrics = pd.DataFrame(first_data, index=[0])
    merged_metrics.loc[0, 'Dataset'] = keys[0]
    index = 1
    for k, v in datasets.items():
        merged_metrics = merged_metrics.append(v, ignore_index=True)
        merged_metrics.loc[index, 'Dataset'] = k
        index = index + 1
    merged_data = merged_metrics.melt(id_vars='Dataset', value_name='values', var_name='metrics')
    return merged_data


# plot functions
def plot_groups_disparity(disparities):
    fig, ax = plt.subplots()
    for d in disparities:
        sns.lineplot(data=d)
    plt.axhline(y=1, linewidth=2)
    ax.set_ylabel('Group disparity')
    return ax


def plot_quality_metrics(q_metrics):
    ax = sns.heatmap(q_metrics, annot=True, cmap=plt.cm.RdBu)
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.yticks(rotation=0)
    plt.title('Classification Report')
    plt.tight_layout()
    return ax


def plot_classification_report(classificationReport,
                               title='Classification report',
                               cmap='RdBu'):
    classificationReport = classificationReport.replace('\n\n', '\n')
    classificationReport = classificationReport.replace(' / ', '/')
    lines = classificationReport.split('\n')

    classes, plotMat, support, class_names = [], [], [], []
    for line in lines[1:-4]:  # if you don't want avg/total result, then change [1:] into [1:-1]
        t = line.strip().split()
        if len(t) < 2:
            continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    plotMat = np.array(plotMat)
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup)
                   for idx, sup in enumerate(support)]

    plt.imshow(plotMat, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    plt.grid(False)
    plt.xticks(np.arange(3), xticklabels, rotation=45)
    plt.yticks(np.arange(len(classes)), yticklabels)

    upper_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 8
    lower_thresh = plotMat.min() + (plotMat.max() - plotMat.min()) / 10 * 2
    for i, j in itertools.product(range(plotMat.shape[0]), range(plotMat.shape[1])):
        plt.text(j, i, format(plotMat[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if (plotMat[i, j] > upper_thresh or plotMat[i, j] < lower_thresh) else "black")

    plt.ylabel('Metrics')
    plt.xlabel('Classes')
    plt.tight_layout()


def plot_metrics_comparison(bias_class_metrics, rw_class_metrics, title1='', title2=''):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    keys = list(bias_class_metrics.keys())
    vals = [float(bias_class_metrics[k]) for k in keys]
    sns.barplot(x=keys, y=vals, ax=ax[0])
    for k in keys:
        ax[0].text(keys.index(k),
                   bias_class_metrics[k],
                   round(bias_class_metrics[k], 3),
                   ha="center",
                   va="bottom",
                   fontsize="medium"
                   )
    keys = list(rw_class_metrics.keys())
    vals = [float(rw_class_metrics[k]) for k in keys]
    sns.barplot(x=keys, y=vals, ax=ax[1])
    for k in keys:
        ax[1].text(keys.index(k),
                   rw_class_metrics[k],
                   round(rw_class_metrics[k], 3),
                   ha="center",
                   va="bottom",
                   fontsize="medium"
                   )
    ax[0].tick_params(labelrotation=90)
    ax[0].set(title=title1)
    ax[1].tick_params(labelrotation=90)
    ax[1].set(title=title2)
    return ax


def plot_syntesis(dataset, title):
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    sns.barplot(data=dataset, x='metrics', y='values', hue='Dataset', ax=ax)
    plt.ylabel(ylabel='')
    plt.xlabel(xlabel='')
    plt.title(title)
    display(dataset.pivot(index='Dataset', values='values', columns='metrics'))
    return ax


def plot_correlation(dataset: BinaryLabelDataset, s: str):
    df = dataset.convert_to_dataframe()[0]
    corr = df.corrwith(df[s], drop=True)
    corr = pd.DataFrame(corr)
    fig, ax = plt.subplots(1, 1, figsize=(10, 1))
    sns.heatmap(corr[(abs(corr[0]) > 0.1)].T, annot=True)
    return ax


# classification and sampling functions

def balance_set(w_exp, w_obs, df, tot_df, round_level=None, debug=False):
    disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
    disparity = [disp]
    while disp != 1:
        if w_exp / w_obs > 1:
            df = df.append(df.sample())
        elif w_exp / w_obs < 1:
            df = df.drop(df.sample().index, axis=0)
        w_obs = len(df) / len(tot_df)
        disp = round(w_exp / w_obs, round_level) if round_level else w_exp / w_obs
        disparity.append(disp)
        if debug:
            print(w_exp / w_obs)
    return df, disparity


def sample_dataset(dataframe: pd.DataFrame,
                   groups_condition: list,
                   fav_label: bool,
                   unfav_label: bool,
                   protected_attribute_names: list,
                   label: str,
                   round_level=None, debug=False):
    df = dataframe.copy()
    groups = [df[cond & fav_label] for cond in groups_condition] + [df[cond & unfav_label] for cond in groups_condition]
    exp_weights = ([(len(df[cond]) / len(df)) * (len(df[fav_label]) / len(df)) for cond in groups_condition] +
                   [(len(df[cond]) / len(df)) * (len(df[unfav_label]) / len(df)) for cond in groups_condition])
    obs_weights = [len(group) / len(df) for group in groups]
    disparities = []
    for i in range(len(groups)):
        groups[i], d = balance_set(exp_weights[i], obs_weights[i], groups[i], df, round_level, debug)
        disparities.append(d)
    df_new = groups.pop().append([group for group in groups]).sample(frac=1)
    print('Original dataset size: (%s,%s)' % dataframe.shape)
    print('Sampled dataset size: (%s,%s)' % df_new.shape)
    plot_groups_disparity(disparities)
    return BinaryLabelDataset(df=df_new, protected_attribute_names=protected_attribute_names, label_names=[label])


def classify(estimator: Pipeline,
             data: BinaryLabelDataset,
             priv_group: list,
             unpriv_group: list,
             sensitive_attributes=None,
             show=True,
             n_splits=10,
             debiaser: Transformer = None,
             ):
    if sensitive_attributes is None:
        sensitive_attributes = []
    np_data = np.hstack((data.features, data.labels))
    kf = KFold(n_splits=n_splits, shuffle=True)
    dataset_metrics = []
    class_metrics = []
    quality_metrics_p = []
    quality_metrics_u = []
    for train, test in kf.split(np_data):
        d_train = data.subset(train)
        d_test = data.subset(test)
        if debiaser:
            d_train = debiaser.fit_transform(d_train)
            d_test = debiaser.transform(d_test)
        x_train, y_train, x_test, y_test = x_y_split(d_train, d_test, sensitive_attributes)
        if sensitive_attributes:
            indexes = [d_train.feature_names.index(s) for s in sensitive_attributes]
            d_train.features = np.delete(d_train.features, indexes, axis=1)
        pipe = deepcopy(estimator)
        pipe.fit(x_train, y_train, logisticregression__sample_weight=d_train.instance_weights.ravel())
        pred = d_test.copy()
        pred.labels = pipe.predict(x_test)
        data_metric = compute_dataset_fairness_metrics(d_train, unpriv_group, priv_group, disp=False)
        metric = compute_fairness_metrics(d_test, pred, unpriv_group, priv_group, disp=False)
        q_metric_p, q_metric_u = compute_quality_metrics(d_test, pred, unpriv_group, priv_group)
        quality_metrics_p.append(q_metric_p)
        quality_metrics_u.append(q_metric_u)
        class_metrics.append(metric)
        dataset_metrics.append(data_metric)

    ris = {key: round(np.mean([metric[key] for metric in class_metrics]), 4) for key in class_metrics[0]}
    q_metrics_p = {key: round(np.mean([metric[key] for metric in quality_metrics_p]), 4) for key in
                   quality_metrics_p[0]}
    q_metrics_u = {key: round(np.mean([metric[key] for metric in quality_metrics_u]), 4) for key in
                   quality_metrics_u[0]}
    q_metrics = pd.DataFrame(data=[q_metrics_p, q_metrics_u], index=['Privileged', 'Unprivileged'])
    d_metrics = {key: round(np.mean([metric[key] for metric in dataset_metrics]), 4) for key in dataset_metrics[0]}
    plot_quality_metrics(q_metrics)
    if show:
        display(Markdown('### Dataset Metrics:'))
        for key, val in d_metrics.items():
            print("%s: %.4f" % (key, val))
        print("\n")
        display(Markdown('### Classification Metrics:'))
        for key, val in ris.items():
            print("%s: %.4f" % (key, val))
    return ris
