# Train a random forest on the Boston Housing dataset and example a sample using our method.
# Produces Figure 5 in the paper.

import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from decomposition_shap.distributions import MultiGaussian
from decomposition_shap.force_dependent import force_dependent_plot
from helpers import figure_folder
from decomposition_shap.kernel_dependent import DependentKernelExplainer

seed = 5
np.random.seed(seed)

# Load data
x, y = load_boston(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit model
forest = xgb.XGBRegressor(n_estimators=1000)
forest.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='rmse', early_stopping_rounds=50)

# Fit distribution
dist = MultiGaussian(x_train)

# Explain
condexp = DependentKernelExplainer(forest.predict, x_train, dist.sample)
sample_index = 16
shaps = condexp.shap_values(x_train[sample_index])

# Plot
data = load_boston()
fig = force_dependent_plot(np.mean(forest.predict(x_train)), shaps,
                           features=x_train[sample_index],
                           feature_names=data['feature_names'], show=False, text_rotation=-90)
fig.savefig(figure_folder + "boston{}.png".format(sample_index), bbox_inches="tight")  # Figure 5 in paper
fig.savefig(figure_folder + "boston{}.pdf".format(sample_index), bbox_inches="tight")
plt.clf()
