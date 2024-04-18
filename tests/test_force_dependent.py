import numpy as np

from decomposition_shap.force_dependent import force_dependent_plot


def test_pos_simple():
    shap_values = np.array([1, 1, 1, 1 / 2, 1 / 2, 1 / 2])
    force_dependent_plot(3, shap_values, features=[0.8, 0.8, 0.8], feature_names=['f1', 'f2', 'f3'])
    assert True


def test_neg_simple():
    shap_values = -np.array([1, 1, 1, 1 / 2, 1 / 2, 1 / 2])
    force_dependent_plot(3, shap_values, features=[0.8, 0.8, 0.8], feature_names=['f1', 'f2', 'f3'])
    assert True


def test_complex():
    shap_values = np.array([-0.1, -1, -1, -2, 2.5, 2, -0.1, -2, -2, -1, 1, -2])
    fig = force_dependent_plot(3, shap_values, features=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                               feature_names=['f1', 'f2', 'f3', 'f4', 'f5', 'f6'],
                               show=False)
    #fig.savefig("dummy.png", bbox_inches="tight")
    assert True


def test_simulated():
    shap_values = np.array([0.4, 0.1, 0.4, 0])
    fig = force_dependent_plot(0.5, shap_values, features=[True, True],
                               feature_names=['prior recidivism/\nhigh heart rate', 'race/\nobesity'],
                               show=False)
    #fig.savefig("sim.png", bbox_inches="tight")
    assert True


def test_logit():
    shap_values = np.array([1, 1, 1, 1 / 2, 1 / 2, 1 / 2])
    force_dependent_plot(3, shap_values, features=[0.8, 0.8, 0.8], feature_names=['f1', 'f2', 'f3'],
                         link='logit')
    assert True
