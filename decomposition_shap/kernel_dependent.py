# Many code parts taken from https://github.com/slundberg/shap/blob/master/shap/explainers
# Hence we included the corresponding MIT License at the bottom of this file.

import numpy as np
import pandas as pd
import scipy as sp
from shap import KernelExplainer
from shap.common import convert_to_instance, match_instance_to_data, convert_to_instance_with_index
from tqdm.auto import tqdm


class DependentKernelExplainer(KernelExplainer):
    """
    Explain output of any function for any situation by decomposing the Shapley value into an
    interventional and dependent effect. Uses Kernel SHAP to compute conditional SHAP values and SamplingExplainer to
    compute interventional effects.
    """

    def __init__(self, model, data, sample, link=None, **kwargs):
        """
        Initialize explainer.
        :param model: function
            User supplied function that takes a matrix of samples (# samples x # features) and
            computes the output of the model for those samples. The output can be a vector
            (# samples) or a matrix (# samples x # model outputs).
        :param data: numpy.array or pandas.DataFrame or shap.common.DenseData or any scipy.sparse matrix
            Background data used only for computing the expected value.
        :param sample: function
            Function which allows to sample the conditional distribution, see distributions.py
        :param link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the ouput with a link function where link(outout) = sum(phi).
        If the model output is a probability then the LogitLink link function makes the feature
        importance values have log-odds units.
        """
        # attribution is constructed completely from the distribution
        if link is None:
            super().__init__(model, data, **kwargs)
        else:
            super().__init__(model, data, link, **kwargs)

        self.sample = sample
        assert str(
            self.link) == "identity", "DependentKernelExplainer currently only supports the identity link not " + str(
            self.link) + " , best to use logodds and use the logit option during plotting with force_dependent_plot"

    def shap_values(self, X, **kwargs):
        """
        Estimate the SHAP values and interventional effects for a set of samples.
        :param X: numpy.array or pandas.DataFrame or any scipy.sparse matrix
            A matrix of samples (# samples x # features) on which to explain the model's output.
        :param neval_kernel: "auto" or int
            Number of model evaluations per permutation to compute conditional SHAP values using Kernel SHAP.
            "auto" means equal to the number samples in the background dataset.
        :param nperm_sampling: "auto" or int
            Number of permutations to sample to compute interventional SHAP effects using SamplingExplainer. Only one
            function evaluation per permutations is used.
            "auto" means equal to 4000 x # varying features
        :return:
            For models with a single output this returns a matrix of SHAP values and interventional effect
            (# samples x (2 x # features)). For models with vector outputs this returns a list
            of such matrices, one for each output.
        """
        neval_kernel = kwargs.get("neval_kernel", "auto")
        if neval_kernel != "auto":
            self.N = neval_kernel

        # Code originality: largely taken from 'shap_values' from KernelExplainer
        # convert dataframes
        if str(type(X)).endswith("pandas.core.series.Series'>"):
            X = X.values
        elif str(type(X)).endswith("'pandas.core.frame.DataFrame'>"):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()
        assert x_type.endswith(arr_type) or sp.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type
        assert len(X.shape) == 1 or len(X.shape) == 2, "Instance must have 1 or 2 dimensions!"

        # single instance
        if len(X.shape) == 1:
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)
            part_explanation = self.explain_part(data, **kwargs)
            # vector-output
            s = explanation.shape
            if len(s) == 2:
                outs = [np.zeros(s[0] * 2) for j in range(s[1])]
                for j in range(s[1]):
                    outs[j][:s[0]] = explanation[:, j]
                    outs[j][s[0]:] = part_explanation[:, j]
                return outs

            # single-output
            else:
                out = np.zeros(s[0] * 2)
                out[:s[0]] = explanation
                out[s[0]:] = part_explanation
                return out

        # explain the whole dataset
        elif len(X.shape) == 2:
            explanations = []
            part_explanations = []
            for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
                data = X[i:i + 1, :]
                if self.keep_index:
                    data = convert_to_instance_with_index(data, column_name, index_value[i:i + 1], index_name)
                explanations.append(self.explain(data, **kwargs))
                # Compute interventional part
                part_explanations.append(self.explain_part(data, **kwargs))

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0] * 2)) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i][:s[0]] = explanations[i][:, j]
                        # Add interventional parts behind it
                        outs[j][i][s[0]:] = part_explanations[i][:, j]
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0] * 2))
                for i in range(X.shape[0]):
                    out[i][:s[0]] = explanations[i]
                    # Add interventional parts behind it
                    out[i][s[0]:] = part_explanations[i]
                return out

    def addsample(self, x, m, w):
        # Code originality: largely taken from 'addsample' from KernelExplainer
        offset = self.nsamplesAdded * self.N
        if isinstance(self.varyingFeatureGroups, (list,)):
            raise NotImplementedError

        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                raise NotImplementedError
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # Here instead of intervening in the background data set with the known features
                # (as is done in the normal SHAP implementation) we make the background data by
                # sampling conditioned on the known features.
                self.synth_data[offset:offset + self.N] = self.sample(features=groups,
                                                                    feature_values=evaluation_data, n=self.N)
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def allocate(self):
        # Code originality: largely taken from 'allocate' from KernelExplainer, note that now the synthetic dataset
        # doesn't matter since we overwrite/make it ourselves by sampling

        # Content doesn't matter, just want the correct shape
        self.synth_data = np.tile(self.data.data, (self.nsamples, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def explain_part(self, incoming_instance, **kwargs):
        # Code originality: almost completely equal to 'explain' from SamplingExplainer,
        # now 'sampling_estimate' = 'sampling_estimate_part'

        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        assert len(self.data.groups) == self.P, "SamplingExplainer does not support feature groups!"

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        # self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingInds)

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values[0]
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then there no feature has an effect
        if self.M == 0:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))
            diff = self.fx - self.fnull
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nperm_sampling = kwargs.get("nperm_sampling", "auto")
            if self.nperm_sampling == "auto":
                self.nperm_sampling = 4000 * self.M

            assert self.nperm_sampling % 2 == 0, "nperm_sampling must be divisible by 2!"
            min_samples_per_feature = kwargs.get("min_samples_per_feature", 100)
            round1_samples = self.nperm_sampling
            round2_samples = 0
            if round1_samples > self.M * min_samples_per_feature:
                round2_samples = round1_samples - self.M * min_samples_per_feature
                round1_samples -= round2_samples

            # divide up the samples among the features for round 1
            nsamples_each1 = np.ones(self.M, dtype=np.int64) * 2 * (round1_samples // (self.M * 2))
            for i in range((round1_samples % (self.M * 2)) // 2):
                nsamples_each1[i] += 2

            # explain every feature in round 1
            phi = np.zeros((self.P, self.D))
            phi_var = np.zeros((self.P, self.D))
            self.X_masked = np.zeros((nsamples_each1.max(), self.data.data.shape[1]))
            for i, ind in enumerate(self.varyingInds):
                phi[ind, :], phi_var[ind, :] = self.sampling_estimate_part(ind, self.model.f, instance.x,
                                                                           self.data.data,
                                                                           nsamples=nsamples_each1[i])

            # optimally allocate samples according to the variance
            if phi_var.sum() == 0:
                phi_var += 1  # spread samples uniformally if we found no variability
            phi_var /= phi_var.sum()
            nsamples_each2 = (phi_var[self.varyingInds, :].mean(1) * round2_samples).astype(np.int)
            for i in range(len(nsamples_each2)):
                if nsamples_each2[i] % 2 == 1: nsamples_each2[i] += 1
            for i in range(len(nsamples_each2)):
                if nsamples_each2.sum() > round2_samples:
                    nsamples_each2[i] -= 2
                elif nsamples_each2.sum() < round2_samples:
                    nsamples_each2[i] += 2
                else:
                    break

            self.X_masked = np.zeros((nsamples_each2.max(), self.data.data.shape[1]))
            for i, ind in enumerate(self.varyingInds):
                if nsamples_each2[i] > 0:
                    val, var = self.sampling_estimate_part(ind, self.model.f, instance.x, self.data.data,
                                                           nsamples=nsamples_each2[i])

                    total_samples = nsamples_each1[i] + nsamples_each2[i]
                    phi[ind, :] = (phi[ind, :] * nsamples_each1[i] + val * nsamples_each2[i]) / total_samples
                    phi_var[ind, :] = (phi_var[ind, :] * nsamples_each1[i] + var * nsamples_each2[i]) / total_samples

            # convert from the variance of the differences to the variance of the mean (phi)
            for i, ind in enumerate(self.varyingInds):
                phi_var[ind, :] /= np.sqrt(nsamples_each1[i] + nsamples_each2[i])

        if phi.shape[1] == 1:
            phi = phi[:, 0]

        return phi

    def sampling_estimate_part(self, j, f, x, X, nsamples=10):
        # Code originality: largely taken from 'sampling_estimate' from SamplingExplainer
        assert nsamples % 2 == 0, "nsamples must be divisible by 2!"
        X_masked = self.X_masked[:nsamples, :]
        inds = np.arange(X.shape[1])

        for i in range(0, nsamples // 2):
            np.random.shuffle(inds)
            pos = np.where(inds == j)[0][0]

            # Here compute 2 terms for interventional SHAP part
            # sample z_unknown knowing z_known
            sample = (self.sample(features=inds[:pos], feature_values=x[0, inds[:pos]], n=1))
            # intervene on z_i
            X_masked[i] = sample
            X_masked[i, inds[pos]] = x[0, inds[pos]]
            # keep z_unknown_i
            X_masked[-i + 1] = sample

        evals = f(X_masked)
        evals_on = evals[:nsamples // 2]
        evals_off = evals[nsamples // 2:][::-1]
        d = evals_on - evals_off

        return np.mean(d, 0), np.var(d, 0)

# Copyright (c) 2018 Scott Lundberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
