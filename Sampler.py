from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer
from IPython.display import display

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


class Sampler(Transformer):
    def __init__(self, round_level=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.round_level = round_level
        self.debug = debug

    def predict(self, dataset):
        return dataset

    def transform(self, dataset):
        return dataset

    def fit_transform(self, dataset: BinaryLabelDataset):
        df = dataset.convert_to_dataframe()[0].copy()
        fav_label = df[dataset.label_names[0]] == dataset.favorable_label
        unfav_label = df[dataset.label_names[0]] == dataset.unfavorable_label

        if len(dataset.protected_attribute_names) == 2:
            s1 = dataset.protected_attribute_names[0]
            s2 = dataset.protected_attribute_names[1]
            groups_condition = [(df[s1] == 0) & (df[s2] == 0), (df[s1] == 1) & (df[s2] == 0),
                                (df[s1] == 0) & (df[s2] == 1), (df[s1] == 1) & (df[s2] == 1)]
        else:
            s1 = dataset.protected_attribute_names[0]
            groups_condition = [(df[s1] == 0), (df[s1] == 1)]
        groups = [df[cond & fav_label] for cond in groups_condition] + [df[cond & unfav_label] for cond in
                                                                        groups_condition]        
        exp_weights = ([(len(df[cond]) / len(df)) * (len(df[fav_label]) / len(df)) for cond in groups_condition] +
                       [(len(df[cond]) / len(df)) * (len(df[unfav_label]) / len(df)) for cond in groups_condition])
        obs_weights = [len(group) / len(df) for group in groups]
        disparities = []
        for i in range(len(groups)):
            groups[i], d = balance_set(exp_weights[i], obs_weights[i], groups[i], df, self.round_level, self.debug)
            disparities.append(d)
        df_new = groups.pop().append([group for group in groups]).sample(frac=1)
        return BinaryLabelDataset(df=df_new,
                                  protected_attribute_names=dataset.protected_attribute_names,
                                  label_names=dataset.label_names)
