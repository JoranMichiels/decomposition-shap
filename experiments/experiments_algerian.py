# Experiment on the Algerian Forest fires dataset.
# Produces plots for Figure 6a and 6b in the paper. Print the values used in Figure 7 and Table 2 in the paper.

import matplotlib
import numpy as np
import pingouin as pg
import xgboost as xgb
from shap import KernelExplainer
from sklearn.model_selection import train_test_split

from decomposition_shap.distributions import GaussianCopula
from decomposition_shap.force_dependent import force_dependent_plot
from force_interventional import force_plot
from helpers import load_algerian, figure_folder
from decomposition_shap.kernel_dependent import DependentKernelExplainer

matplotlib.rcdefaults()

x, y = load_algerian(return_X_y=True)
data = load_algerian()

seed = 11

np.random.seed(seed)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = xgb.XGBClassifier(n_estimators=1000)
model.fit(x_train, y_train, eval_set=[(x_test, y_test)])

dist = GaussianCopula(x)

condexp = DependentKernelExplainer(lambda d: model.predict(d, output_margin=True), x, dist.sample)

intexp = KernelExplainer(lambda d: model.predict(d, output_margin=True), x)

# Plot a sample and compare it with interventional SHAP values

sample_index = 32
fig = force_dependent_plot(np.mean(model.predict(x, output_margin=True)), condexp.shap_values(x_test[sample_index]),
                           features=x_test[sample_index],
                           feature_names=list(data.columns)[:-1], show=False, link='logit')
fig.savefig(figure_folder + "algerian{}.png".format(sample_index), bbox_inches="tight")  # Figure 6a in paper
fig.savefig(figure_folder + "algerian{}.pdf".format(sample_index), bbox_inches="tight")

int_fig = force_plot(np.mean(model.predict(x, output_margin=True)), intexp.shap_values(x_test[sample_index]),
                     features=x_test[sample_index],
                     feature_names=list(data.columns)[:-1], show=False, link='logit')
int_fig.savefig(figure_folder + "int_algerian{}.png".format(sample_index), bbox_inches="tight")  # Figure 6b in paper
int_fig.savefig(figure_folder + "int_algerian{}.pdf".format(sample_index), bbox_inches="tight")

# Correlation analysis

y_pred = model.predict(x, output_margin=True)

data['Prediction'] = y_pred
data = data.drop(columns='Classes')

vals_cond_x = condexp.shap_values(x)

nb_features = 4

dependent_part = vals_cond_x[:, :nb_features] - vals_cond_x[:, nb_features:]
int_part = vals_cond_x[:, nb_features:]

# Values used in Figure 6 in paper
for feature1 in data.columns:
    for feature2 in data.columns:
        if feature2 != feature1:
            par_cor = pg.partial_corr(data, feature1, feature2,
                                      [f for f in data.columns if f != feature1 and f != feature2], method='spearman')

            print(feature1, feature2, par_cor.iloc[0]['r'])

dependent_corr = []
int_corr = []
for i in range(nb_features):
    dependent_corr += [pg.corr(dependent_part[:, i], data.iloc[:, i], method='spearman').iloc[0]['r']]
    int_corr += [pg.corr(int_part[:, i], data.iloc[:, i], method='spearman').iloc[0]['r']]

# Values used in Table 2 of paper
print('dep_corr', dependent_corr)
print('int_corr', int_corr)
