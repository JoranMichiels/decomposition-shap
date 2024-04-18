import os

import pandas as pd


def get_file_path(file):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, file)


save_folder = get_file_path('checkpoints/')
figure_folder = get_file_path('figures/')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)


def load_algerian(return_X_y=False):
    data = pd.read_csv(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00547/Algerian_forest_fires_dataset_UPDATE.csv',
        usecols=['Temperature', 'RH', 'Ws', 'Rain', 'Classes'], skiprows=[0, 124, 125, 126], sep=' *, *')
    # Fix some errors in the dataset
    data.Classes = data.Classes.str.strip()
    data.at[165, 'Classes'] = 'fire'

    data.Classes.replace(('fire', 'not fire'), (True, False), inplace=True)
    if return_X_y:
        return data.iloc[:, :-1].values, data.iloc[:, -1].values
    else:
        return data
