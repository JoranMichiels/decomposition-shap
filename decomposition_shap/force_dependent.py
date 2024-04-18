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


def logit(x):
    return 1 / (1 + np.exp(-x))


def force_dependent_plot(base_value, shap_values, features=None, feature_names=None, out_names=None, link="identity",
                         show=False, figsize=(17, 4),
                         text_rotation=0):
    """
    Visualize the given SHAP values and interventional effects in a matplotlib force plot
    :param base_value: float
        This is the reference value that the feature contributions start from.
    :param shap_values: numpy.array
        Conditional SHAP values and interventional effects together in one array (2 x # features).
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

    :return: plot if show=False, else nothing.
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
        "outValue": np.sum(shap_values[0, :len(features)]) + base_value,
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

    # Split and sort parts
    model_effects = shap_values[0, len(features):]
    dep_effects = shap_values[0, :len(features)] - shap_values[0, len(features):]

    try:
        neg_names = neg_features[:, 2]
        if len(neg_names) == len(feature_names):
            pos_names = []
        else:
            pos_names = pos_features[:, 2]

    except IndexError:
        neg_names = []
        pos_names = pos_features[:, 2]

    sorter = np.argsort(feature_names)
    neg_indices = sorter[np.searchsorted(feature_names, neg_names, sorter=sorter)]
    pos_indices = sorter[np.searchsorted(feature_names, pos_names, sorter=sorter)]

    total_effect = np.abs(total_neg) + total_pos

    # Create partial bars for negative shap values
    rectangle_list = draw_all_parts(out_value, model_effects[neg_indices], dep_effects[neg_indices], neg_features,
                                    'negative', width_bar / 2, vertical_offset=-1.5 * width_bar,
                                    total_effect=total_effect, min_perc=0.05)

    for i in rectangle_list:
        ax.add_patch(i)

    # Create partial bars for positive shap values
    rectangle_list = draw_all_parts(out_value, model_effects[pos_indices], dep_effects[pos_indices], pos_features,
                                    'positive', width_bar / 2, vertical_offset=-1.5 * width_bar,
                                    total_effect=total_effect, min_perc=0.05)

    for i in rectangle_list:
        ax.add_patch(i)

    # Add labels
    fig, ax = draw_labels_regions(fig, ax, out_value, model_effects[neg_indices], dep_effects[neg_indices],
                                  neg_features, 'negative',
                                  vertical_offset_part=-1.5 * width_bar, width_part=width_bar / 2,
                                  vertical_offset_text=-1.5 * width_bar, offset_text=offset_text,
                                  total_effect=total_effect, min_perc=0.05,
                                  text_rotation=text_rotation)

    fig, ax = draw_labels_regions(fig, ax, out_value, model_effects[pos_indices], dep_effects[pos_indices],
                                  pos_features, 'positive',
                                  vertical_offset_part=-1.5 * width_bar, width_part=width_bar / 2,
                                  vertical_offset_text=-1.5 * width_bar, offset_text=offset_text,
                                  total_effect=total_effect, min_perc=0.05,
                                  text_rotation=text_rotation)

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
    draw_model_dependent_distinction(fig, right_pos, -1.5 * width_bar - width_bar / 2, ax)

    plt.rc('xtick', labelsize=14)
    plt.locator_params(axis='x', nbins=8)
    if show:
        plt.show()
    else:
        return plt.gcf()


def draw_all_parts(out_value, model_effects, dep_effects, features, feature_type, width_bar,
                   vertical_offset, total_effect, min_perc=0.05):
    """Draw both model and dependent parts for particular feature type"""
    rectangle_list = []

    pre_val = out_value
    for index, model_effect, dep_effect, feature in zip(range(len(model_effects)), model_effects, dep_effects,
                                                        features):
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break
        if model_effect * dep_effect < 0:
            # scale effects
            factor = np.abs(model_effect + dep_effect) / (np.abs(model_effect) + np.abs(dep_effect))
            model_effect *= factor
            dep_effect *= factor
        if feature_type == 'positive':
            if model_effect < 0:
                model_right_bound = pre_val
                model_left_bound = pre_val - np.abs(model_effect)
                dep_right_bound = model_left_bound
                dep_left_bound = float(feature[0])
                pre_val = dep_left_bound
            else:
                dep_right_bound = pre_val
                dep_left_bound = pre_val - np.abs(dep_effect)
                model_right_bound = dep_left_bound
                model_left_bound = float(feature[0])
                pre_val = model_left_bound
        else:
            if model_effect > 0:
                model_left_bound = pre_val
                model_right_bound = pre_val + np.abs(model_effect)
                dep_left_bound = model_right_bound
                dep_right_bound = float(feature[0])
                pre_val = dep_right_bound
            else:
                dep_left_bound = pre_val
                dep_right_bound = pre_val + np.abs(dep_effect)
                model_left_bound = dep_right_bound
                model_right_bound = float(feature[0])
                pre_val = model_right_bound

        # Create model rectangle
        model_points_rectangle = [[model_left_bound, vertical_offset],
                                  [model_right_bound, vertical_offset],
                                  [model_right_bound, -width_bar + vertical_offset],
                                  [model_left_bound, -width_bar + vertical_offset]]
        dep_points_rectangle = [[dep_left_bound, -width_bar + vertical_offset],
                                [dep_right_bound, -width_bar + vertical_offset],
                                [dep_right_bound, -2 * width_bar + vertical_offset],
                                [dep_left_bound, -2 * width_bar + vertical_offset]]

        model_line = plt.Polygon(model_points_rectangle, closed=True, fill=True,
                                 facecolor='#FF0D57' if model_effect > 0 else '#1E88E5', linewidth=0)
        dep_line = plt.Polygon(dep_points_rectangle, closed=True, fill=True,
                               facecolor='#FF0D57' if dep_effect > 0 else '#1E88E5', linewidth=0)
        rectangle_list += [model_line, dep_line]

    return rectangle_list


def draw_labels_regions(fig, ax, out_value, model_effects, dep_effects, features, feature_type, offset_text,
                        vertical_offset_part, width_part, vertical_offset_text, total_effect=0, min_perc=0.05,
                        text_rotation=0):
    start_text = out_value
    pre_val = out_value
    total_text_offset = vertical_offset_part - 2 * width_part + vertical_offset_text

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
    if feature_type == 'positive' and len(model_effects) > 0 > model_effects[0] * dep_effects[0]:
        x, y = np.array([[pre_val, pre_val], [0, -0.18]])
        line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    # Initial shade path
    path = [[out_value, 0], [out_value, total_text_offset - 0.05]]

    box_end = out_value
    val = out_value
    for i, feature in enumerate(features):
        # Exclude all labels that do not contribute at least 10% to the total
        feature_contribution = np.abs(float(feature[0]) - pre_val) / np.abs(total_effect)
        if feature_contribution < min_perc:
            break

        # Compute value for current feature
        val = float(feature[0])

        model_effect = model_effects[i]
        dep_effect = dep_effects[i]

        # Draw labels.
        if feature[1] == "":
            text = feature[2]
        else:
            text = feature[2] + ' = ' + feature[1]

        if text_rotation != 0:
            va_alignment = 'top'
        else:
            va_alignment = 'baseline'

        text_out_val = plt.text(start_text - sign * offset_text,
                                total_text_offset, text,
                                fontsize=15, color=colors[0],
                                horizontalalignment=alignement,
                                va=va_alignment,
                                rotation=text_rotation,
                                ma='center')
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

        # Create end line
        if model_effect * dep_effect < 0:
            # scale effects
            factor = np.abs(model_effect + dep_effect) / (np.abs(model_effect) + np.abs(dep_effect))
            model_effect *= factor
            dep_effect *= factor
            neg_ef, pos_ef = sorted([model_effect, dep_effect])
            vert_offsets = [0, vertical_offset_part, vertical_offset_part - 2 * width_part,
                            total_text_offset + 0.03,
                            total_text_offset - 0.05]

            if (sign * box_end_) > (sign * val):
                box_end = val
            else:
                box_end = box_end_ - sign * offset_text

            if sign == 1:
                x, y = np.array([[pre_val, pre_val + neg_ef, pre_val + neg_ef, start_text, start_text],
                                 vert_offsets])
                line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
                line.set_clip_on(False)
                ax.add_line(line)

                x, y = np.array([[val, val + np.abs(neg_ef), val + np.abs(neg_ef), box_end, box_end],
                                 vert_offsets])
                line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
                line.set_clip_on(False)
                ax.add_line(line)
                path.extend(
                    map(list, zip([pre_val, pre_val + neg_ef, pre_val + neg_ef, start_text, start_text], vert_offsets)))
                path.extend(map(list, reversed(
                    list(zip([val, val + np.abs(neg_ef), val + np.abs(neg_ef), box_end, box_end], vert_offsets)))))
                path += [[box_end, vert_offsets[-1]]]
            if sign == -1:
                x, y = np.array([[pre_val, pre_val + pos_ef, pre_val + pos_ef, start_text, start_text],
                                 vert_offsets])
                line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
                line.set_clip_on(False)
                ax.add_line(line)

                x, y = np.array([[val, val - pos_ef, val - pos_ef, box_end, box_end],
                                 vert_offsets])
                line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
                line.set_clip_on(False)
                ax.add_line(line)
                path.extend(
                    map(list, zip([pre_val, pre_val + pos_ef, pre_val + pos_ef, start_text, start_text], vert_offsets)))
                path.extend(
                    map(list, reversed(list(zip([val, val - pos_ef, val - pos_ef, box_end, box_end], vert_offsets)))))
                path += [[box_end, vert_offsets[-1]]]

            start_text = box_end

        elif (sign * box_end_) > (sign * val):
            x, y = np.array([[val, val], [0, total_text_offset - 0.05]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val
            path += [[val, total_text_offset - 0.05]]

        else:
            box_end = box_end_ - sign * offset_text
            x, y = np.array([[val, val, box_end, box_end],
                             [0, vertical_offset_part - 2 * width_part, total_text_offset + 0.03,
                              total_text_offset - 0.05]])
            line = lines.Line2D(x, y, lw=1., alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end
            path += [[box_end, total_text_offset - 0.05]]

        # Update previous value
        pre_val = float(feature[0])

    # Create line for labels
    extent_shading = [out_value, box_end, 0, total_text_offset - 0.35]
    path += [[box_end, total_text_offset + 0.03], [pre_val, vertical_offset_part - 2 * width_part], [pre_val, 0],
             [out_value, 0]]
    path = Path(path)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    # If the feature goes over the side of the plot, we remove that label
    # and stop drawing labels
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


def draw_output_element(out_name, out_value, ax, link):
    # Add output value
    x, y = np.array([[out_value, out_value], [0, 0.24]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight('bold')
    if link == 'logit':
        link = logit
    else:
        link = lambda x: x
    text_out_val = plt.text(out_value, 0.30, '{0:.2f}'.format(link(out_value)),
                            fontproperties=font,
                            fontsize=14,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

    text_out_val = plt.text(out_value, 0.43, out_name,
                            fontsize=18, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))


def draw_model_dependent_distinction(fig, right_bound, div_position, ax):
    text_out_val = plt.text(right_bound, div_position, 'interventional ',
                            fontsize=18, alpha=0.5,
                            ha='right', va='bottom')
    plt.text(right_bound, div_position, 'dependent ',
             fontsize=18, alpha=0.5,
             ha='right', va='top')
    # We need to draw the plot to be able to get the size of the
    # text box
    fig.canvas.draw()
    text_out_val.set_bbox(dict(facecolor='none', edgecolor='none'))
    box_size = text_out_val.get_bbox_patch().get_extents() \
        .transformed(ax.transData.inverted())
    x, y = np.array([[right_bound, box_size.get_points()[1][0]], [div_position, div_position]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)


def draw_higher_lower_element(out_value, offset_text):
    plt.text(out_value - offset_text, 0.555, 'higher ',
             fontsize=18, color='#FF0D57',
             horizontalalignment='right')

    plt.text(out_value + offset_text, 0.555, ' lower',
             fontsize=18, color='#1E88E5',
             horizontalalignment='left')

    plt.text(out_value, 0.55, r'$\leftarrow$',
             fontsize=18, color='#1E88E5',
             horizontalalignment='center')

    plt.text(out_value, 0.575, r'$\rightarrow$',
             fontsize=18, color='#FF0D57',
             horizontalalignment='center')


def draw_base_element(base_value, ax):
    x, y = np.array([[base_value, base_value], [0.13, 0.25]])
    line = lines.Line2D(x, y, lw=2., color='#F2F2F2')
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = plt.text(base_value, 0.33, 'base value',
                            fontsize=18, alpha=0.5,
                            horizontalalignment='center')
    text_out_val.set_bbox(dict(facecolor='white', edgecolor='white'))

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
