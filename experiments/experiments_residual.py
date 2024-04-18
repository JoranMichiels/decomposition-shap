# Experiment comparing the dependent SHAP part with Shapley Residuals.
# Produces Figure 4 in the paper.

import matplotlib.pyplot as plt
import numpy as np

from helpers import figure_folder

# Correlation plot comparison
coef = 2


def dependent(alpha):
    return 0.5 * (1 + coef) * alpha


def norm_residual(alpha):
    return np.sqrt(2) * np.abs(0.5 * coef - (1 + 0.5 * coef) * alpha)


alphas = np.linspace(0, 1, 200)
plt.plot(alphas, dependent(alphas), label="Dependent SHAP part")
plt.plot(alphas, norm_residual(alphas), label="Norm Shapley residual")
plt.xlabel('Correlation coefficient')
plt.ylabel('Dependency attribution')
plt.legend()
plt.savefig(figure_folder + 'correlation.png', dpi=400)
plt.savefig(figure_folder + 'correlation.pdf')
