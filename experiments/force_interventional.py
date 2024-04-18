# Big parts taken/inspired from shap package 'force_matplotlib.py' and 'force.py'
# Hence we included the corresponding MIT License at the bottom of this file.

import warnings

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import lines
    from matplotlib.font_manager import FontProperties
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    import matplotlib
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass

from shap.plots.force_matplotlib import format_data, update_axis_limits, draw_bars
from decomposition_shap.force_dependent import draw_output_element, draw_base_element, logit, draw_higher_lower_element


def force_plot(base_value, shap_values, features=None, feature_names=None, out_names=None, link="identity",
               show=False, figsize=(17, 4),
               text_rotation=0):
    """
    Visualize the given SHAP values in a matplotlib force plot. (This fixes some errors and provides extra features wrt
    the original matplotlib force plot implementation)
    :param base_value: float
        This is the reference value that the feature contributions start from.
    :param shap_values: numpy.array
        The SHAP values (# features)
    :param features: numpy.array
        Matrix of feature values (# features)
    :param feature_names: numpy.array
        List of feature names (# features).
    :param out_names: str
        The name of the output of the model (plural to support multi-output plotting in the future).
    :param link: "identity" or "logit"
        The transformation used when drawing the tick mark labels. Using logit will add a second axis with
        probabilities from the log-odds numbers.
    :param show: bool
        Whether to show or return the plot.
    :param figsize:
        Size of the matplotlib figure.
    :param text_rotation: float
        Rotation of the labels: 0 is horizontal, -90 is vertical.

    :return: plt.gcf() if show=False, else nothing.
    """

    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (1, len(shap_values)))

    # Make data for original code
    features_dict = {}

    for i in range(len(features)):
        features_dict[i] = {'effect': shap_values[0, i],
                            'value': features[i]}

    data = {
        "outNames": ['model output value'] if out_names is None else out_names,
        "baseValue": base_value,
        "outValue": np.sum(shap_values[0]) + base_value,
        "link": 'identity',  # keep values in logodds
        "featureNames": feature_names,
        "features": features_dict
    }

    # Turn off interactive plot
    if show == False:
        plt.ioff()

    # Format data
    neg_features, total_neg, pos_features, total_pos = format_data(data)

    # Compute overall metrics
    base_value = data['baseValue']
    out_value = data['outValue']
    offset_text = (np.abs(total_neg) + np.abs(total_pos)) * 0.04

    # Define plots
    fig, ax = plt.subplots(figsize=figsize)

    # Compute axis limit
    update_axis_limits(ax, total_pos, pos_features, total_neg,
                       neg_features, base_value)

    if link == 'logit':
        # Set secondary axis in probability space, keep original figure for easy relative comparison of features
        ax.tick_params(axis='x', direction='in', pad=-17)
        ax.secondary_xaxis('top', functions=(logit, lambda p: np.log(p) - np.log(1 - p)))

    # Define width of bar
    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    # Create bar for negative shap values
    rectangle_list, separator_list = draw_bars(out_value, neg_features, 'negative',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    # Create bar for positive shap values
    rectangle_list, separator_list = draw_bars(out_value, pos_features, 'positive',
                                               width_separators, width_bar)
    for i in rectangle_list:
        ax.add_patch(i)

    for i in separator_list:
        ax.add_patch(i)

    total_effect = np.abs(total_neg) + total_pos

    # Add labels
    fig, ax = draw_labels(fig, ax, out_value, neg_features, 'negative',
                          offset_text, total_effect, min_perc=0.05, text_rotation=text_rotation)

    fig, ax = draw_labels(fig, ax, out_value, pos_features, 'positive',
                          offset_text, total_effect, min_perc=0.05, text_rotation=text_rotation)

    # higher lower legend
    draw_higher_lower_element(out_value, offset_text)

    # Add label for base value
    draw_base_element(base_value, ax)

    # Add output label
    out_names = data['outNames'][0]
    draw_output_element(out_names, out_value, ax, link=link)

    # Add divider
    if len(pos_features) == 0:
        right_pos = out_value
    else:
        right_pos = float(pos_features[-1][0])

    plt.rc('xtick', labelsize=14)
    plt.locator_params(axis='x', nbins=8)

    if show:
        plt.show()
    else:
        return plt.gcf()


def draw_labels(fig, ax, out_value, features, feature_type, offset_text, total_effect=0, min_perc=0.05,
                text_rotation=0):
    start_text = out_value
    pre_val = out_value

    # Define variables specific to positive and negative effect features
    if feature_type == 'positive':
        colors = ['#FF0D57', '#FFC3D5']
        alignement = 'right'
        sign = 1
    else:
        colors = ['#1E88E5', '#D1E6FA']
        alignement = 'left'
        sign = -1

    # Draw initial line
    if feature_type == 'positive':
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for feature in features:
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        # Compute value for current feature
        val = float(feature[0])

        # Draw labels.
        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + ' = ' + feature[1]

        if text_rotation is not 0:
            va_alignment = 'top'
        else:
            va_alignment = 'baseline'

        text_out_val = plt.text(start_text - sign * offset_text,
                                -0.15, text,
                                fontsize=15, color=colors[0],
                                horizontalalignment=alignement,
                                va=va_alignment,
                                rotation=text_rotation)
        text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))

        # We need to draw the plot to be able to get the size of the
        # text box
        fig.canvas.draw()
        box_size = text_out_val.get_bbox_patch().get_extents() \
            .transformed(ax.transData.inverted())
        if feature_type == 'positive':
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]

        # If the feature goes over the side of the plot, we remove that label
        # and stop drawing labels
        if box_end_ > ax.get_xlim()[1]:
            text_out_val.remove()
            break

        # Create end line
        if (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, box_end, box_end],
                             [0, -0.08, -0.18]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        # Update previous value
        pre_val = float(feature[0])

    # Create line for labels
    extent_shading = [out_value, box_end, 0, -0.31]
    path = [[out_value, 0], [pre_val, 0], [box_end, -0.08],
            [box_end, -0.2], [out_value, -0.2],
            [out_value, 0]]

    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    # Extend axis if needed
    lower_lim, upper_lim = ax.get_xlim()
    if (box_end < lower_lim):
        ax.set_xlim(box_end, upper_lim)

    if (box_end > upper_lim):
        ax.set_xlim(lower_lim, box_end)

    # Create shading
    if feature_type == 'positive':
        colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.
    else:
        colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.

    cm = matplotlib.colors.LinearSegmentedColormap.from_list('cm', colors)

    _, Z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    im = plt.imshow(Z2, interpolation='quadric', cmap=cm,
                    vmax=0.01, alpha=0.3,
                    origin='lower', extent=extent_shading,
                    clip_path=patch, clip_on=True, aspect='auto')
    im.set_clip_path(patch)

    return fig, ax

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
