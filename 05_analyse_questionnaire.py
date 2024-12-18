#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import matplotlib
from matplotlib import pyplot
import numpy
import pingouin
import scipy.stats


# # # # #
# SETUP

# BASIC CONTROLS
# Names for data folders and files.
DATAFOLDER = "data_exp_163666-v4_2024-08-21"
# Names of the questionnaires (the version we produced by loading and parsing 
# the Gorilla output).
QUESTIONNAIRES = [ \
    "questionnaire_parent_status.csv", \
    "questionnaire_pdds.csv", \
    "questionnaire_dsr.csv", \
    ]

# Average both, or use only the last presentation to compute average avoidance.
# The first presentation is subject to morbid curiosity, so might not 
# accurately reflect the strongest avoidance tendencies.
USE_SECOND_PRESENTATION = False
    
# ANALYSIS SETTINGS
# Affective stimuli for each condition.
STIMULI = { \
    "control": [ \
        "control_01_disgust", \
        "control_02_disgust_1131_body_products_lum", \
        "control_03_disgust_1138_body_products_lum", \
        "control_04_disgust_1120_body_products_lum", \
        "control_05_disgust", \
        ], \

    "meconium": [ \
        "meconium_01_disgust_J2023-05-05_19-07-45", \
        "meconium_02_disgust_I2020-12-13_21-00-51", \
        ], \

    "milk": [ \
        "milk_01_disgust_I2020-12-19_04-48-12", \
        "milk_02_disgust_I2020-12-22_13-48-05", \
        "milk_03_disgust_I2021-02-18_08-26-27", \
        "milk_04_disgust_I2021-01-25_10-11-41", \
        "milk_05_disgust_J2023-10-06_09-27-45", \
        ], \

    "weaning": [ \
        "weaning_01_disgust_I2021-04-14_11-01-34", \
        "weaning_02_disgust_I2021-04-25_18-12-10", \
        "weaning_03_disgust_I2021-05-13_09-40-23", \
        "weaning_04_disgust_I2021-06-17_12-51-29", \
        "weaning_05_disgust_I2021-09-07_07-36-38", \
        "weaning_06_disgust_I2021-11-14_06-49-12", \
        "weaning_07_disgust_I2021-11-16_10-36-18", \
        "weaning_08_disgust_I2022-03-24_06-38-26", \
        "weaning_09_disgust_I2022-07-07_07-34-24", \
        "weaning_10_disgust_I2022-10-23_07-04-51", \
        "weaning_11_disgust_I2022-12-08_07-37-44", \
        "weaning_12_disgust_I2023-01-12_07-33-18", \
        "weaning_13_disgust_I2023-09-09_15-09-40", \
        ], \
    }

# Nappy frequency question options.
NAPPY_FREQ_OPTIONS = ["None of my children wear nappies", "Never", \
    "Less than once", "1", "2", "3-5", "6-10", "11-15", "16-20", "21-30", \
    "More than 30"]

# Conditions.
CONDITIONS = list(STIMULI.keys())
CONDITIONS.sort()
# Trial duration in milliseconds.
TRIAL_DURATION = 10000
# Bin size for re-referencing samples to bins across the trial duration.
# This is in milliseconds.
BINWIDTH = 100.0 / 3.0

# FILES AND FOLDERS
# Path the this file's directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data directory and files within it.
DATADIR = os.path.join(DIR, "data", DATAFOLDER)
PPNAMESPATH = os.path.join(DATADIR, "ppnames.dat")
PPGROUPSPATH = os.path.join(DATADIR, "ppgroups.dat")
WEANINGPATH = os.path.join(DATADIR, "weaning_status.dat")
DWELLPATH = os.path.join(DATADIR, "dwell_mouse_memmap.dat")
DWELLSHAPEPATH = os.path.join(DATADIR, "dwell_mouse_shape_memmap.dat")
if not os.path.isdir(DATADIR):
    raise Exception("ERROR: Could not find data directory at '{}'".format( \
        DATADIR))
# Path to the output directory and files within it.
OUTDIR = os.path.join(DIR, "output_{}".format(DATAFOLDER))
if not os.path.isdir(OUTDIR):
    raise Exception("ERROR: Could not find output directory at '{}'".format( \
        OUTDIR))

# PLOTTING
PLOTCOLS = { \
    "groups":{ \
        "control":  "#2e3436", \
        "milk":     "#204a87", \
        "weaned":   "#a40000", \
        }, \
    }

PLOTCOLMAPS = { \
    "control":  "Greens", \
    "meconium": "copper_r", \
    "milk":     "Wistia", \
    "weaning":  "YlOrBr", \
    }

PLOTLINESTYLES = {"Man":"--", "Woman":"-"}


# # # # #
# LOAD DATA

# LOAD QUESTIONNAIRE DATA\
data_questionnaires = None
for questionnaire_file in QUESTIONNAIRES:
    # Load the file.
    fpath = os.path.join(OUTDIR, questionnaire_file)
    if os.path.isfile(fpath):
        with open(fpath, "r") as f:
            header = f.readline().replace("\n", "").split(",")
        raw = numpy.loadtxt(fpath, dtype=str, delimiter=",", skiprows=1, \
            unpack=True)
        data_ = {}
        for i, var in enumerate(header):
            try:
                data_[var] = raw[i,:].astype(numpy.float64)
            except:
                data_[var] = raw[i,:]

        if data_questionnaires is None:
            data_questionnaires = {}
            for var in data_.keys():
                data_questionnaires[var] = data_[var]
        else:
            for var in data_.keys():
                if var == "ppname":
                    continue
                data_questionnaires[var] = numpy.zeros( \
                    data_questionnaires["ppname"].shape, \
                    dtype=data_[var].dtype)
                missing = numpy.ones( \
                    data_questionnaires["ppname"].shape, dtype=bool)
                for i, ppname in enumerate(data_questionnaires["ppname"]):
                    j = list(data_["ppname"]).index(ppname)
                    data_questionnaires[var][i] = data_[var][j]
                    missing[i] = False
                try:
                    data_questionnaires[var][missing] = numpy.nan
                except:
                    data_questionnaires[var][missing] = "nan"
            

# LOAD MEMMAP DATA
# Load participant names and group membership.
ppnames = numpy.memmap(PPNAMESPATH, dtype="<U24", mode="r")
ppgroups = numpy.memmap(PPGROUPSPATH, dtype="<U11", mode="r")
weaning_status = numpy.memmap(WEANINGPATH, dtype="<U7", mode="r")
# Load the dwell data's shape from the dwell_shape file.
dwell_shape = tuple(numpy.memmap(DWELLSHAPEPATH, dtype=numpy.int32, \
    mode="r"))
# Load the dwell data from file.
dwell = numpy.memmap(DWELLPATH, dtype=numpy.float32, mode="r", \
    shape=dwell_shape)
# Recompute the bin edges.
bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
    dtype=numpy.float32)


# # # # #
# COMPUTE AVOIDANCE SCORES

avoid = {}

for ci, condition in enumerate(CONDITIONS):
    
    # Average over bins.
    # d then has shape (n_participants, n_stimuli, n_presentations, n_aoi)
    d = numpy.nanmean(dwell[:,ci,:,:,:,:], axis=4)
    
    # Compute the difference between neutral and affective stimulus. This
    # quantifies approach (positive values) or avoidance (negative values).
    # Computed as affective minus neutral, only for the current condition.
    # d then has shape (n_participants, n_stimuli, n_presentations)
    d = d[:,:,:,0] - d[:,:,:,1]
    
    # Average over all different stimuli, and recode as percentages.
    # val then has shape (n_participants, n_presentations)
    val = 100 * numpy.nansum(d, axis=1) / len(STIMULI[condition])
    
    # Compute the mean over presentations.
    avoid[condition] = numpy.nanmean(val, axis=1)


# # # # #
# COMPUTE QUESTIONNAIRE SCORES

quest = {}

# Compute questionnaire sum and average scores.
for questionnaire in ["dsr", "pdds"]:
    for stat in ["count", "sum"]:
        quest["{}_{}".format(questionnaire, stat)] = \
            numpy.zeros(len(ppnames), dtype=float)
    for var in data_questionnaires.keys():
        if questionnaire in var:
            if "catch" in var:
                pass
            else:
                for i, ppname in enumerate(ppnames):
                    ppi = list(data_questionnaires["ppname"]).index(ppname)
                    quest["{}_count".format(questionnaire)][i] += 1
                    quest["{}_sum".format(questionnaire)][i] += \
                        data_questionnaires[var][ppi]

    quest["{}_avg".format(questionnaire)] = \
        quest["{}_sum".format(questionnaire)] \
        / quest["{}_count".format(questionnaire)]

# Copy a few bits of data over from the parent status questionnaire.
for var in ["number_of_children", "child_age_months", "gender", \
    "nappy_change-frequency", "nappy_change"]:
    if var in ["child_age_months", "nappy_change"]:
        dtype = float
    else:
        dtype = data_questionnaires[var].dtype
    quest[var] = numpy.zeros(len(ppnames), dtype=dtype)
    for i, ppname in enumerate(ppnames):
        ppi = list(data_questionnaires["ppname"]).index(ppname)
        if var == "child_age_months":
            quest[var][i] = data_questionnaires["age_months_only_child"][ppi]
            if numpy.isnan(quest[var][i]):
                quest[var][i] = data_questionnaires["age_months_youngest"][ppi]
        elif var == "nappy_change":
            quest[var][i] = (data_questionnaires["nappy_change_wee"][ppi] \
                + data_questionnaires["nappy_change_poo"][ppi]) / 2.0
        else:
            quest[var][i] = data_questionnaires[var][ppi]


# # # # #
# NAPPY CHANGE FREQUENCY COUNTS

with open(os.path.join(OUTDIR, "nappy_change_freq.tsv"), "w") as f:
    
    header = ["gender", "stage"] + NAPPY_FREQ_OPTIONS
    f.write("\t".join(map(str, header)))
    
    weaning_stages = numpy.unique(weaning_status)
    for gender in ["all", "Man", "Woman"]:
        
        for ws in weaning_stages:
            sel = weaning_status == ws
            if gender in ["Man", "Woman"]:
                sel = sel & (quest["gender"] == gender)
            line = [gender, ws]
            for i, option in enumerate(NAPPY_FREQ_OPTIONS):
                count = numpy.sum(quest["nappy_change-frequency"][sel]==i)
                line.append(count)
            
            f.write("\n" + "\t".join(map(str, line)))


# # # # #
# SENSITIVITY STATS

with open(os.path.join(OUTDIR, "dsr_stats.tsv"), "w") as f:
    
    header = ["variable", "gender", "stage", "m", "sd", "m_control", \
        "sd_control", "t", "df", "p", "d", "BF10", "BF01"]
    f.write("\t".join(map(str, header)))
    
weaning_stages = {"control":["control"], "milk":["wean_0"], \
    "weaned":["wean_1", "wean_2", "wean_3"]}

variables = ["dsr_avg", "pdds_avg"]
for var in variables:
    for gender in ["all", "Man", "Woman"]:
        
        for ws in ["milk", "weaned"]:
            sel_con = numpy.isin(weaning_status, weaning_stages["control"])
            sel = numpy.isin(weaning_status, weaning_stages[ws])
            if gender in ["Man", "Woman"]:
                sel_con = sel_con & (quest["gender"] == gender)
                sel = sel & (quest["gender"] == gender)
            
            result = pingouin.ttest(quest[var][sel], quest[var][sel_con], \
                paired=False, alternative="two-sided", correction=True, \
                r=0.707)

            m = numpy.nanmedian(quest[var][sel])
            sd = numpy.nanstd(quest[var][sel])
            m_con = numpy.nanmedian(quest[var][sel_con])
            sd_con = numpy.nanstd(quest[var][sel_con])

            with open(os.path.join(OUTDIR, "dsr_stats.tsv"), "a") as f:
                line = [var, gender, ws, m, sd, m_con, sd_con, \
                    result["T"][0], result["dof"][0], result["p-val"][0], \
                    result["cohen-d"][0], result["BF10"][0], \
                    1.0/float(result["BF10"][0])]
                f.write("\n" + "\t".join(map(str, line)))


# # # # #
# AVOIDANCE PLOTS

for gender_split in [True, False]:

    # Create a new figure.
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.07, top=0.99)
    
    # Get the group names.
    group_names = numpy.unique(ppgroups)
    group_names.sort()
    # Set x positions for the conditions.
    x_pos = range(1, len(CONDITIONS)+1)
    # Set x offset per group.
    x_offset = numpy.linspace(-0.25, 0.25, len(group_names))
    x_width = 0.75 * (x_offset[1] - x_offset[0])
    # Compute limits on x-axis.
    xlim = (-x_width+x_pos[0]+x_offset[0], x_width+x_pos[-1]+x_offset[-1])
    
    # Plot the separation line and background colours.
    x_line = numpy.array(xlim)
    ax.plot(x_line, numpy.zeros(x_line.shape), ':', lw=3, color="black", alpha=0.5)
    ax.fill_between(x_line, numpy.ones(x_line.shape)*-100, \
        numpy.zeros(x_line.shape), color="black", alpha=0.1)
    annotate_x = x_pos[0] + x_offset[0] - x_width + 0.02*(x_line[-1]-x_line[0])
    ax.annotate("Disgust approach", (annotate_x, 100-10), fontsize=12, alpha=0.5)
    ax.annotate("Disgust avoidance", (annotate_x, -100+5), fontsize=12, alpha=0.5)
    
    # Loop through conditions.
    x_labels = []
    legend_distributions = []
    legend_labels = []
    for ci, condition in enumerate(CONDITIONS):
        
        if condition == "control":
            x_labels.append("Bodily effluvia")
        else:
            x_labels.append("{} diaper".format(condition.capitalize()))
        
        # Average over bins.
        # d then has shape (n_participants, n_stimuli, n_presentations, n_aoi)
        d = numpy.nanmean(dwell[:,ci,:,:,:,:], axis=4)
        
        # Compute the difference between neutral and affective stimulus. This
        # quantifies approach (positive values) or avoidance (negative values).
        # Computed as affective minus neutral, only for the current condition.
        # d then has shape (n_participants, n_stimuli, n_presentations)
        d = 100 * (d[:,:,:,0] - d[:,:,:,1])
        
        # Average over stimuli.
        # d then has shape (n_participants, n_repetitions)
        d = numpy.nansum(d, axis=1) / len(STIMULI[condition])
        
        # Average over repetitions.
        # d then has shape (n_participants, n_repetitions)
        if USE_SECOND_PRESENTATION:
            d = d[:,1]
        else:
            d = numpy.nanmean(d, axis=1)
        
        # Plot for every group.
        for gi, group in enumerate(group_names):
            
            sel = ppgroups == group

            if not gender_split:
                distributions = ax.violinplot(dataset=d[sel], \
                    positions=[x_pos[ci]+x_offset[gi]], widths=[x_width], \
                    showmeans=False, showmedians=False, showextrema=False)
                boxes = ax.boxplot(d[sel], positions=[x_pos[ci]+x_offset[gi]], \
                    widths=[x_width*0.4])
                distributions["bodies"][0].set_color(PLOTCOLS["groups"][group])
                boxes["medians"][0].set_color(PLOTCOLS["groups"][group])
                boxes["medians"][0].set_linewidth(3)

            # Draw box plots for men and women separately.
            else:
                for gei, gender in enumerate(["Man", "Woman"]):
                    gender_offset = [-x_width/3, x_width/3]
                    # Draw violin plot.
                    distributions = ax.violinplot( \
                        dataset=d[sel][quest["gender"][sel]==gender], \
                        positions=[x_pos[ci]+x_offset[gi]+gender_offset[gei]], \
                        widths=[x_width*0.5], \
                        showmeans=False, showmedians=False, showextrema=False)
                    distributions["bodies"][0].set_color(PLOTCOLS["groups"][group])
                    # Draw box plot.
                    boxes = ax.boxplot(d[sel][quest["gender"][sel]==gender], \
                        positions=[x_pos[ci]+x_offset[gi]+gender_offset[gei]], \
                        widths=[x_width*0.4*0.5])
                    boxes["medians"][0].set_color(PLOTCOLS["groups"][group])
                    boxes["medians"][0].set_linewidth(3)
                    for element in ["whiskers", "boxes"]:
                        for line in boxes[element]:
                            line.set_linestyle(PLOTLINESTYLES[gender])

            # Add the violin body colour to the legend list, but only on the first
            # pass through so that we don't overpopulate the lists with double 
            # entires.
            if ci == 0:
                legend_distributions.append(distributions["bodies"][0])
                legend_labels.append("{} group".format(group.capitalize()))
                if gender_split and (gi == len(group_names)-1):
                    for gender in ["Man", "Woman"]:
                        legend_distributions.append(matplotlib.lines.Line2D( \
                            [0], [0], color="#000000", lw=3, \
                            ls=PLOTLINESTYLES[gender]))
                        legend_labels.append( \
                            {"Man":"Men", "Woman":"Women"}[gender])
    
    # Finish the plot.
    ax.legend(legend_distributions, legend_labels, loc="upper right", fontsize=14)
    ax.set_ylabel("Dwell difference (% pt.)", fontsize=16)
    ax.set_ylim(-100, 100)
    # ax.set_xlabel("Stimulus type", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlim(xlim)
    # Save the figure.
    if gender_split:
        fpath = os.path.join(OUTDIR, \
            "disgust_avoidance_per_condition_per_group_gender.png")
    else:
        fpath = os.path.join(OUTDIR, \
            "disgust_avoidance_per_condition_per_group.png")
    fig.savefig(fpath)
    pyplot.close(fig)


# # # # #
# QUESTIONNAIRE PLOTS

for gender_split in [True, False]:

    # Create a new figure.
    fig, ax = pyplot.subplots(nrows=1, ncols=1, figsize=(8.0, 6.0), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.07, top=0.99)
    
    # Set the variables that should be plotted.
    variables = ["dsr_avg", "pdds_avg", "nappy_change"]
    variable_labels = { \
        "dsr_avg":          "General", \
        "pdds_avg":         "Parental", \
        "nappy_change":     "Diaper", 
        }
    
    # Get the group names.
    group_names = numpy.unique(ppgroups)
    group_names.sort()
    # Set x positions for the conditions.
    x_pos = range(1, len(variables)+1)
    # Set x offset per group.
    x_offset = numpy.linspace(-0.25, 0.25, len(group_names))
    x_width = 0.75 * (x_offset[1] - x_offset[0])
    # Compute limits on x-axis.
    xlim = (-x_width+x_pos[0]+x_offset[0], x_width+x_pos[-1]+x_offset[-1])
    
    # Loop through conditions.
    x_labels = []
    legend_distributions = []
    legend_labels = []
    for vi, var in enumerate(variables):
        
        # Get the label for this variable.
        x_labels.append(variable_labels[var])
        
        # Get the data for this variable.
        d = quest[var]
        
        # Plot for every group.
        for gi, group in enumerate(group_names):
            
            sel = (ppgroups == group) & numpy.invert(numpy.isnan(d))
            if numpy.sum(sel) == 0:
                continue

            if not gender_split:
                # Draw violin plot.
                distributions = ax.violinplot(dataset=d[sel], \
                    positions=[x_pos[vi]+x_offset[gi]], widths=[x_width], \
                    showmeans=False, showmedians=False, showextrema=False)
                distributions["bodies"][0].set_color(PLOTCOLS["groups"][group])
                # Draw box plot.
                boxes = ax.boxplot(d[sel], positions=[x_pos[vi]+x_offset[gi]], \
                    widths=[x_width*0.4])
                boxes["medians"][0].set_color(PLOTCOLS["groups"][group])
                boxes["medians"][0].set_linewidth(3)

            # Draw box plots for men and women separately.
            else:
                for gei, gender in enumerate(["Man", "Woman"]):
                    gender_offset = [-x_width/3, x_width/3]
                    # Draw violin plot.
                    distributions = ax.violinplot( \
                        dataset=d[sel][quest["gender"][sel]==gender], \
                        positions=[x_pos[vi]+x_offset[gi]+gender_offset[gei]], \
                        widths=[x_width*0.5], \
                        showmeans=False, showmedians=False, showextrema=False)
                    distributions["bodies"][0].set_color(PLOTCOLS["groups"][group])
                    # Draw box plot.
                    boxes = ax.boxplot(d[sel][quest["gender"][sel]==gender], \
                        positions=[x_pos[vi]+x_offset[gi]+gender_offset[gei]], \
                        widths=[x_width*0.4*0.5])
                    boxes["medians"][0].set_color(PLOTCOLS["groups"][group])
                    boxes["medians"][0].set_linewidth(3)
                    for element in ["whiskers", "boxes"]:
                        for line in boxes[element]:
                            line.set_linestyle(PLOTLINESTYLES[gender])

            # Add the violin body colour to the legend list, but only on the first
            # pass through so that we don't overpopulate the lists with double 
            # entires.
            if vi == 0:
                legend_distributions.append(distributions["bodies"][0])
                legend_labels.append("{} group".format(group.capitalize()))
                if gender_split and (gi == len(group_names)-1):
                    for gender in ["Man", "Woman"]:
                        legend_distributions.append(matplotlib.lines.Line2D( \
                            [0], [0], color="#000000", lw=3, \
                            ls=PLOTLINESTYLES[gender]))
                        legend_labels.append( \
                            {"Man":"Men", "Woman":"Women"}[gender])
    
    # Finish the plot.
    ax.legend(legend_distributions, legend_labels, loc="upper right", fontsize=14)
    ax.set_ylabel("Disgust sensitivity (average)", fontsize=16)
    ax.set_ylim(0, 4)
    # ax.set_xlabel("Stimulus type", fontsize=16)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=14)
    ax.set_xlim(xlim)
    # Save the figure.
    if gender_split:
        fpath = os.path.join(OUTDIR, "disgust_sensitivity_per_group_gender.png")
    else:
        fpath = os.path.join(OUTDIR, "disgust_sensitivity_per_group.png")
    fig.savefig(fpath)
    pyplot.close(fig)

