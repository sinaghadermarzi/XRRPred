from decimal import Decimal

import scipy.stats
import numpy
from matplotlib import pyplot
import csv
import os
import random
from itertools import combinations
import copy
import matplotlib

myfont = {'family': 'Times New Roman',
          'weight': 'normal',
          'size': 11}


# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

def add_values(bp, ax):
    """ This actually adds the numbers to the various points of the boxplots"""
    for element in ['whiskers', 'medians', 'caps']:
        for line in bp[element]:
            # Get the position of the element. y is the label you want
            (x_l, y), (x_r, _) = line.get_xydata()
            # Make sure datapoints exist
            # (I've been working with intervals, should not be problem for this case)
            if not numpy.isnan(y):
                x_line_center = x_l + (x_r - x_l) / 2
                y_line_center = y  # Since it's a line and it's horisontal
                # overlay the value:  on the line, from center to right
                ax.text(x_line_center, y_line_center + 0.01,  # Position
                        '%.3f' % y,  # Value (3f = 3 decimal float)
                        verticalalignment='center',  # Centered vertically with line
                        fontsize=4, backgroundcolor=(0, 0, 0, 0))


pyplot.rc('font', **myfont)
pyplot.rc('axes', edgecolor=(0, 0, 0, 0.2), linewidth=1)


def my_format(number):
    if (abs(number) < 0.001) or (abs(number) > 9999):
        return '{:.1E}'.format(number)
    else:
        return '{:.3F}'.format(number)


median_str = ''


def do_stat(formatted_data, set_names, rng=None,name ="stat"):
    global median_str
    # is_binary = False
    # if is_binary

    # first we create the box plot containing all the sets
    #    formatted_data= []
    labels = set_names
    is_normal = {}
    # set_names = list(data.keys())

    for i in range(len(set_names)):
        #        labels.append(set_name)
        #        formatted_data.append(data[set_name][var_name])
        #        #for each of the sets we check for normality and take note of it in is_normal
        set_name = set_names[i]
        results = scipy.stats.anderson(numpy.array(formatted_data[i]))
        is_normal[set_name] = (results.statistic < results.critical_values[2])
    width = len(formatted_data)/2
    fig = pyplot.figure(random.randint(1, 100), figsize=(width, 5))
    # color = [dict(boxes='blue'),dict(boxes='green'),dict(boxes='orange'),dict(boxes='cyan')]
    wp = dict(linewidth=1)
    # # fp = dict(marker='o', markerfacecolor='green', markersize=12,
    #               linestyle='none')
    cp = dict(linewidth=1)
    bp = dict(linewidth=1)
    mp = dict(linewidth=1, color="black")
    meanpointprops = dict(marker='X', markeredgecolor='black', markerfacecolor='black')
    ax = fig.add_subplot(111)
    bp = ax.boxplot(formatted_data, medianprops=mp, boxprops=bp, capprops=cp, whiskerprops=wp, labels=labels,
                    showfliers=False, patch_artist=True, whis=[5, 95], widths=0.7, showmeans=True,
                    meanprops=meanpointprops)
    colors = [(30 / 255, 113 / 255, 69 / 255), (153 / 255, 180 / 255, 51 / 255), (0, 163 / 255, 0),
              (218 / 255, 83 / 255, 44 / 255), (255 / 255, 196 / 255, 13 / 255), (130 / 255, 90 / 255, 44 / 255)]
    ax.tick_params(axis='x', colors=(1, 1, 1),rotation=90)
    ax.tick_params(axis='y', colors=(0, 0, 0, 1), which='major', direction='inout', labelsize=9)
    ax.tick_params(axis='y', colors=(0, 0, 0, 1), which='minor', direction='inout')

    # ax.set_xlabel("x-label",colors=(0, 0, 0,1))
    # ax.set_ylabel("y-label",colors=(0, 0, 0, 1))
    # for patch, color in zip(bp['boxes'], colors):
    #     patch.set_facecolor(color)
    # [t.set_color('red') for t in ax.yaxis.get_ticklabels()]
    ax.grid(b=True, which='major', axis='y', linewidth=0.8, linestyle='--')
    ax.grid(b=True, which='minor', axis='y', linewidth=0.3, linestyle='--')
    ax.grid(b=True, which='major', axis='x', linewidth=0.3, linestyle='--')
    ax.minorticks_on()
    pyplot.setp(ax.get_yticklabels(), color=(0, 0, 0, 1))
    if (rng == None):
        top = -float("inf")
        bot = -top
        for arr in formatted_data:
            curr_top = numpy.percentile(arr, 95)
            curr_bot = numpy.percentile(arr, 5)
            top = max(top, curr_top)
            bot = min(bot, curr_bot)
        y_range = top - bot
        top += y_range * 0.15
    else:
        bot = rng[0]
        top = rng[1]
        y_range = top - bot
    # bot -= y_range*0.025
    ax.set_ylim(bot, top)
    ax.set(xlim=(0, width*2 +1))
    ax.tick_params(axis='both', which='both', colors=(0, 0, 0, 1), labelsize=12)
    ax.set_xticklabels(labels, ha='center', color=(0, 0, 0, 1))
    # ax.tick_params(axis='both', which='major', labelsize=11)

    # ax.set_yticklabels(ax.get_yticklabels(),color=(0, 0, 0,1))
    ax.xaxis.set_minor_locator(pyplot.NullLocator())
    # ax.yaxis.set_major_formatter(pyplot.colo)
    # ax.yaxis.set_major_locator(pyplot.NullLocator())
    # ax.xaxis.set_major_locator(pyplot.NullLocator())
    # ax.xaxis.set_major_locator(pyplot.AutoLocator())
    fig.tight_layout()
    # add_values(bp,ax)
    title_str = " "
    # pyplot.title(title_str)

    #####################################
    #####################################
    #####################################
    ########an unstructural code to put p-values on the whiskers
    # if True:
    #     idx_s = 0
    #     target = []
    #     for line in bp["caps"]:
    #         patch_str = ''
    #         if (idx_s % 2 == 1):
    #             idx = idx_s // 2
    #             for i in range(0, len(set_names)):
    #                 if i != idx:
    #                     #         add a line for the p-value
    #                     A = set_names[idx]
    #                     B = set_names[i]
    #                     if is_normal[A] and is_normal[B]:
    #                         # if both of them are normally distributed, we use t-test
    #
    #                         test_stat, p_val = scipy.stats.ttest_ind(formatted_data[idx], formatted_data[i])
    #                         if p_val < 0.05 and p_val >= 0.0001:
    #                             patch_str += B + "*\n"
    #                         elif p_val < 0.0001:
    #                             patch_str += B + "**\n"
    #                     else:
    #                         # else we do Anderson-Darling rank-sum test
    #                         t_stat, p_val = scipy.stats.ranksums(formatted_data[idx], formatted_data[i])
    #                         if p_val < 0.05 and p_val >= 0.0001:
    #                             patch_str += B + "*\n"
    #                         elif p_val < 0.0001:
    #                             patch_str += B + "**\n"
    #             patch_str = patch_str[:-1]
    #         idx_s += 1
    #         num_lines = len(patch_str.splitlines())
    #
    #         # Get the position of the element. y is the label you want
    #         (x_l, y), (x_r, _) = line.get_xydata()
    #         # Make sure datapoints exist
    #         # (I've been working with intervals, should not be problem for this case)
    #         if not numpy.isnan(y):
    #             # overlay the value:  on the line, from center to right
    #             ax.text(x_l - 0.07, y + y_range * 0.015 * num_lines,  # Position
    #                     patch_str,  # Value (3f = 3 decimal float)
    #                     verticalalignment='center', fontsize=8,
    #                     bbox=dict(pad=0, fc='white', ec='none'))  # Centered vertically with line
    #             # fontsize=4, backgroundcolor="white")
    #
    # ########end of put p_values in the fig
    # #####################################
    # #####################################
    # #####################################

    if not os.path.exists('./stat_out'):
        os.mkdir('./stat_out')
    pyplot.savefig("stat_out/"+name+".svg", dpi=600, whis=[5, 95], format='svg')
    pyplot.close()

    # comb = combinations(set_names, 2)
    # # writing out statistics for the sets
    # stat_res = "Stats for sets\n"
    # set_stat_str = ""
    # set_stat_str += "Percentiles/set names\t"
    #
    # for i in range(len(set_names)):
    #     set_stat_str += "\t" + set_names[i]
    #
    # set_stat_str += "\n5%\t"
    # for i in range(len(set_names)):
    #     p = numpy.percentile(formatted_data[i], 5)
    #     set_stat_str += "\t" + "{0:.3f}".format(Decimal(str(p)))
    # set_stat_str += "\n25%\t"
    # for i in range(len(set_names)):
    #     p = numpy.percentile(formatted_data[i], 25)
    #     set_stat_str += "\t" + "{0:.3f}".format(Decimal(str(p)))
    # set_stat_str += "\n50%\t"
    # for i in range(len(set_names)):
    #     p = numpy.median(formatted_data[i])
    #     set_stat_str += "\t" + "{0:.3f}".format(Decimal(str(p)))
    # set_stat_str += "\n75%\t"
    # for i in range(len(set_names)):
    #     p = numpy.percentile(formatted_data[i], 75)
    #     set_stat_str += "\t" + "{0:.3f}".format(Decimal(str(p)))
    # set_stat_str += "\n95%\t"
    # for i in range(len(set_names)):
    #     p = numpy.percentile(formatted_data[i], 95)
    #     set_stat_str += "\t" + "{0:.3f}".format(Decimal(str(p)))
    # set_stat_str += "\nDistr\t"
    # for set_name in set_names:
    #     if is_normal[set_name]:
    #         set_stat_str += "\tNorm"
    #     else:
    #         set_stat_str += "\tNotNorm"
    # #    if median_str =='':
    # #        for set_name in set_names:
    # #            median_str += set_name + "\t"
    # #        median_str+='\n'
    #
    # #    median_str+=var_name+'\t'
    # #    for set_name in set_names:
    # #        med_set = numpy.median(data[set_name][var_name])
    # #        med_D = numpy.median(data["D"][var_name])
    # #        percD25 = numpy.percentile(data["D"][var_name],25)
    # #        percD75 = numpy.percentile(data["D"][var_name],75)
    # #        rng_D = percD75-percD25
    # #        median_str += str((med_set-med_D)/rng_D) + "\t"
    # #    median_str += '\n'
    #
    # set_stat_str += "\n\n\nSignificances of difference between sets"
    # for pair in comb:
    #     A, B = pair
    #     i = set_names.index(A)
    #     j = set_names.index(B)
    #     if is_normal[A] and is_normal[B]:
    #         # if both of them are normally distributed, we use t-test
    #         test_stat, p_val = scipy.stats.ttest_ind(formatted_data[i], formatted_data[j])
    #         set_stat_str += "\n" + A + "-" + B + "[T-test p-value]:\t" + "{:.2E}".format(p_val)
    #     else:
    #         # else we do Anderson-Darling rank-sum test
    #         t_stat, p_val = scipy.stats.ranksums(formatted_data[i], formatted_data[j])
    #         set_stat_str += "\n" + A + "-" + B + "[Ranksum p-value]:\t" + "{0:.2E}".format(p_val)
    # if not os.path.exists('./stat_out'):
    #     os.mkdir('./stat_out')
    # with open('./stat_out/stats.txt', 'w') as txtfile:
    #     txtfile.writelines(set_stat_str)


