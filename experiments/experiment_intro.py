# Simulated example from introduction (recidivism vs heart attack)
# Produces Figure 2 in the paper

import numpy as np

from decomposition_shap.force_dependent import force_dependent_plot
from helpers import figure_folder

# In this case we can compute the exact values
shap_values = np.array([0.4, 0.1, 0.4, 0])
fig = force_dependent_plot(0.5, shap_values, features=[True, True],
                           feature_names=['prior recidivism/\nhigh heart rate', 'race/\nobesity'], show=False)
fig.savefig(figure_folder + "sim.png", bbox_inches="tight", dpi=400)  # Figure 2 in paper
fig.savefig(figure_folder + "sim.pdf", bbox_inches="tight")
