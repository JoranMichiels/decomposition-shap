import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from decomposition_shap.distributions import MultiGaussian
from decomposition_shap.force_dependent import force_dependent_plot
from decomposition_shap.kernel_dependent import DependentKernelExplainer

np.random.seed(0)

# Load Breast Cancer Wisconsin dataset
x, y = load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fit a model
model = xgb.XGBClassifier(n_estimators=10)
model.fit(x_train, y_train)

# Fit feature distribution (assumed Multivariate Gaussian)
dist = MultiGaussian(x_train)
# Note: custom distributions can also be defined see decomposition_shap/distribution.py

# Initialize explainer
explainer = DependentKernelExplainer(lambda d: model.predict(d, output_margin=True), x_train, dist.sample)
# Note: as with regular SHAP, usually it makes more sense to explain the logodds 'model.predict(d, output_margin=True)'

# Explain the fifth test sample
sample_index = 3
shaps = explainer.shap_values(x_test[sample_index])  # returns array of conditional SHAP values + interventional effects

# Plot the sample
data = load_breast_cancer()
# With link='logit' the logodds are shown under the axis, and the probability is shown above the axis
fig = force_dependent_plot(np.mean(model.predict(x_train)), shaps, features=x_test[sample_index],
                           feature_names=data['feature_names'], link='logit', text_rotation=-90)  # returns plt.gcf()
fig.savefig("example.png", bbox_inches="tight")
