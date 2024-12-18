#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import copy

import numpy
import matplotlib
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from MouseViewParser.readers import gorilla


# # # # #
# CONSTANTS

# BASIC CONTROLS
# Names for data folders and files.
DATAFOLDER = "data_exp_163666-v4_2024-08-21"
FILENAME = "data_exp_163666-v4_task-bwmk.csv"
FILEFOLDER = "uploads"
# Name of the parent status questionnaire (the version we produced by loading
# and parsing the Gorilla output).
PARENT_STATUS_FILE = "questionnaire_parent_status.csv"

# Overwrite the temporary data?
OVERWRITE_TMP = True

# Create individual trial plots?
CREATE_TRIAL_PLOTS = False

# EXPERIMENT SETTINGS
# Specify the custom fields that should be loaded from the data.
CUSTOM_FIELDS = ["left_stim", "right_stim", "condition", "side"]
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
# Size of the images in pixels.
STIM_SHAPE = (1024, 768)
# Create a flat list of all affective stimuli (used to check stimulus names
# against without having to faff around with keys in the STIMULI dict).
AFFECTIVE_STIMULI = []
for condition in STIMULI.keys():
    AFFECTIVE_STIMULI += STIMULI[condition]

# Conditions.
CONDITIONS = list(STIMULI.keys())
CONDITIONS.sort()
# Areas of interest.
AOI = ["affective", "neutral", "other"]
# Number of trials per condition.
NTRIALS = 25
# Number of trials per stimulus (affective stimuli get repeated).
NTRIALSPERSTIM = 2
# Trial duration in milliseconds.
TRIAL_DURATION = 10000
# Expected sampling rate (Hz).
SAMPLE_RATE = 60.0

# ANALYSIS SETTINGS
# Bin size for re-referencing samples to bins across the trial duration.
# This is in milliseconds.
BINWIDTH = 100.0 / 3.0

# EXCLUSIONS
# Excluded participants.
EXCLUDED = [ \
    ]

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
STIMDIR = os.path.join(DIR, "stimuli")
DATADIR = os.path.join(DIR, "data", DATAFOLDER)
TRIALDATADIR = os.path.join(DATADIR, FILEFOLDER)
FILEPATH = os.path.join(DATADIR, FILENAME)
PPNAMESPATH = os.path.join(DATADIR, "ppnames.dat")
PPGROUPSPATH = os.path.join(DATADIR, "ppgroups.dat")
WEANINGPATH = os.path.join(DATADIR, "weaning_status.dat")
DWELLPATH = os.path.join(DATADIR, "dwell_mouse_memmap.dat")
DWELLSHAPEPATH = os.path.join(DATADIR, "dwell_mouse_shape_memmap.dat")
SCANPATH = os.path.join(DATADIR, "scan_mouse_memmap.dat")
SCANSHAPEPATH = os.path.join(DATADIR, "scan_mouse_shape_memmap.dat")
OUTDIR = os.path.join(DIR, "output_{}".format(DATAFOLDER))
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)
if CREATE_TRIAL_PLOTS:
    IOUTDIR = os.path.join(OUTDIR, "trials")
    if not os.path.isdir(IOUTDIR):
        os.mkdir(IOUTDIR)

# PLOTTING
PLOTCOLMAPS = { \
    "control":  "Greens", \
    "meconium": "copper_r", \
    "milk":     "Wistia", \
    "weaning":  "YlOrBr", \
    }
# Font sizes for different elements.
FONTSIZE = { \
    "title":            32, \
    "axtitle":          24, \
    "legend":           14, \
    "bar":              14, \
    "label":            16, \
    "ticklabels":       12, \
    "annotation":       14, \
    }
# Set the y limits for various plots.
YLIM = {"dwell_p":  [-60, 40]}
# Set the locations of legends.
LEGENDLOCS = { \
    "control":  "lower right", \
    "meconium": "lower right", \
    "milk":     "lower right", \
    "weaning":  "lower right", \
    }


# # # # #
# LOAD DATA

# Load the parent status file.
fpath = os.path.join(OUTDIR, PARENT_STATUS_FILE)
if os.path.isfile(fpath):
    with open(fpath, "r") as f:
        header = f.readline().replace("\n", "").split(",")
    raw = numpy.loadtxt(fpath, dtype=str, delimiter=",", skiprows=1, \
        unpack=True)
    parent_status = {}
    for i, var in enumerate(header):
        try:
            parent_status[var] = raw[i,:].astype(numpy.float64)
        except:
            parent_status[var] = raw[i,:]

    # Use info from the parent status questionnaire to define weaning status.
    parent_status["group"] = numpy.zeros(parent_status["ppname"].shape, dtype="<U11")
    # Participants with 0 children are in the control group.
    sel = parent_status["number_of_children"]==0
    parent_status["group"][sel] = "control"
    # Participants with 1 child who is not yet weaning are in the milk group.
    sel = (parent_status["number_of_children"]==1) & (parent_status["weaning_youngest"]==0)
    parent_status["group"][sel] = "milk"
    # Participants with only pre-weaning children are in the milk group.
    sel = (parent_status["weaning_youngest"]==0) & (parent_status["weaning_any"]==0)
    parent_status["group"][sel] = "milk"
    # Participants with a youngest pre-weaning child are in the milk group, 
    # even if they also have older weaning/weaned children.
    sel = (parent_status["weaning_youngest"]==0) & (parent_status["weaning_any"]==1)
    parent_status["group"][sel] = "milk"
    # Participants with no pre-weaning children are in the weaned group.
    sel = (parent_status["weaning_youngest"]>0) \
        | ((parent_status["weaning_youngest"]>0) & (parent_status["weaning_any"]==1))
    parent_status["group"][sel] = "weaned"


# Load the raw data.
if (not os.path.isfile(DWELLPATH)) or OVERWRITE_TMP:
    
    print("Loading raw data...")

    # Load the file.
    data = gorilla.read_file(FILEPATH, TRIALDATADIR, \
        custom_fields=CUSTOM_FIELDS, use_public_id=True, verbose=False)

    # Count the number of participants.
    participants = list(data.keys())
    participants.sort()
    n_participants = len(participants)
    
    print("Raw data containts {} participants".format(n_participants))
    
    # Exclusions.
    for participant in EXCLUDED:
        if participant in participants:
            print("Excluding '{}'".format(participant))
            data.pop(participant)
            participants.remove(participant)
    
    # Recount after exclusions.
    n_participants = len(participants)
    print("\nIncluding {} participants".format(n_participants))
    
    # Create memmaps for the participant names and their groups.
    ppnames = numpy.memmap(PPNAMESPATH, dtype="<U24", mode="w+", \
        shape=len(participants))
    ppnames[:] = participants
    ppgroups = numpy.memmap(PPGROUPSPATH, dtype="<U11", mode="w+", \
        shape=len(participants))
    weaning_status = numpy.memmap(WEANINGPATH, dtype="<U7", mode="w+", \
        shape=len(participants))
    for i, ppname in enumerate(participants):
        psi = numpy.where(parent_status["ppname"] == ppname)[0][0]
        ppgroups[i] = parent_status["group"][psi]
        if numpy.isnan(parent_status["weaning_youngest"][psi]):
            ws = "control"
        else:
            ws = "wean_{}".format(int(parent_status["weaning_youngest"][psi]))
        weaning_status[i] = ws

    # Compute a few numbers that are relevant for determining the shape of 
    # data matrices.
    n_conditions = len(CONDITIONS)
    n_stimuli_per_condition = max([len(STIMULI[condition]) for condition in CONDITIONS])
    n_aois = len(AOI) - 1
    n_bins = int(numpy.ceil(TRIAL_DURATION / BINWIDTH))
    n_samples = int(numpy.ceil((TRIAL_DURATION / 1000.0) * SAMPLE_RATE))
    # Empty matrix for dwell time to start with.
    shape = numpy.array([n_participants, n_conditions, \
        n_stimuli_per_condition, NTRIALSPERSTIM, n_aois, n_bins], \
        dtype=numpy.int32)
    dwell_shape = numpy.memmap(DWELLSHAPEPATH, dtype=numpy.int32, \
        mode="w+", shape=(len(shape)))
    dwell_shape[:] = shape[:]
    dwell_shape.flush()
    dwell = numpy.memmap(DWELLPATH, dtype=numpy.float32, mode="w+", \
        shape=tuple(shape))
    dwell[:] = 0.0
    # Empty matrix for scan paths to start with.
    shape = numpy.array([n_participants, n_conditions, \
        n_stimuli_per_condition, NTRIALSPERSTIM, 2, n_samples], \
        dtype=numpy.int32)
    scan_shape = numpy.memmap(SCANSHAPEPATH, dtype=numpy.int32, \
        mode="w+", shape=(len(shape)))
    scan_shape[:] = shape[:]
    scan_shape.flush()
    scan = numpy.memmap(SCANPATH, dtype=numpy.int16, mode="w+", \
        shape=tuple(shape))
    scan[:] = 0
    
    # Compute the bin edges.
    bin_edges = numpy.arange(0, TRIAL_DURATION+BINWIDTH/2, BINWIDTH, \
        dtype=numpy.float32)
    # Compute the expected sample times.
    t_samples = numpy.linspace(0, (1000.0/SAMPLE_RATE)*(n_samples-1), \
        n_samples)
    
    # Run through all participants.
    print("\nLooping through participants...")
    for pi, participant in enumerate(participants):
        
        print("\tProcessing participant '{}' ({}/{})".format( \
            participant, pi+1, len(participants)))
        
        # Construct the path to a trial directory for this participant.
        if CREATE_TRIAL_PLOTS:
            trial_outdir = os.path.join(IOUTDIR, participant)
            if not os.path.isdir(trial_outdir):
                os.mkdir(trial_outdir)

        # Extract the participant's display resolution and view rect.
        dispsize = list(map(int, data[participant]["resolution"].split("x")))
        viewrect = list(map(int, data[participant]["viewport"].split("x")))
        
        # Keep track of the count for each affective stimulus.
        stim_count = {}
    
        # Run through all trials.
        for i in range(len(data[participant]["trials"])):
    
            # Start with some blank variables.
            condition = None
            stimuli = {"affective":None, "neutral":None}
            affective_stim = None
            aoi_rect = {"left_stim":None, "right_stim":None}
    
            # Run through all messages in this trial.
            for j, msg in enumerate(data[participant]["trials"][i]["msg"]):
                    
                # Process the message that contains the condition info.
                if msg[0] == "condition":
                    condition = msg[1]
    
                # Process the messages related to the stimulus.
                elif msg[0] in ["left_stim", "right_stim"]:
                    # Split the extension and the file name.
                    name, ext = os.path.splitext(msg[1])
                    if name in AFFECTIVE_STIMULI:
                        stimuli["affective"] = name
                        affective_stim = msg[0]
                    else:
                        stimuli["neutral"] = name
                
                # Process the AOI rect messages.
                elif type(msg[0]) == float:
                    # Parse the message.
                    msg_type, msg_zone, msg_x, msg_y, msg_w, msg_h = \
                        msg[1].split(";")
                    if ("Zone1" in msg_zone) or ("Zone2" in msg_zone):
                        # Extract the rect.
                        zrect = [ \
                            msg_x.replace("zone_x=", ""), \
                            msg_y.replace("zone_y=", ""), \
                            msg_w.replace("zone_w=", ""), \
                            msg_h.replace("zone_h=", ""), \
                            ]
                        zrect = map(float, zrect)
                        zrect = map(round, zrect)
                        zrect = map(int, zrect)
                        zrect = list(zrect)
                        # Convert zone rect (space allocated for image to
                        # appear within) to the presented image rect (actual
                        # image rect). Images are presented in the centre of
                        # the zone rect. They are not scaled if the zone rect
                        # exceeds the image shape, but they are shrunk down if 
                        # the image width exceeds the zone rect's width. In
                        # this case, images are scaled so that the presented
                        # image width is the zone rect width. (The image 
                        # retains its aspect ratio.)
                        zrect_centre = (zrect[0] + zrect[2]//2, \
                            zrect[1] + zrect[3]//2)
                        # If the stimulus fits within the zone rect, retain
                        # its original shape.
                        if STIM_SHAPE[0] <= zrect[2]:
                            stim_shape = copy.deepcopy(STIM_SHAPE)
                        # If the stimulus is larger than the rect, scale it
                        # down so that it fits within the rect's width.
                        else:
                            shrink_p = zrect[2] / float(STIM_SHAPE[0])
                            stim_shape = (round(STIM_SHAPE[0] * shrink_p), \
                                round(STIM_SHAPE[1] * shrink_p))
                        # Compute the rect of the stimulus as it was presented.
                        rect = (zrect_centre[0]-stim_shape[0]//2, \
                            zrect_centre[1]-stim_shape[1]//2, \
                            stim_shape[0], stim_shape[1])
                        # Store the image rect.
                        if "Zone1" in msg_zone:
                            aoi_rect["left_stim"] = copy.deepcopy(rect)
                        elif "Zone2" in msg_zone:
                            aoi_rect["right_stim"] = copy.deepcopy(rect)
            
            # Skip the practice trial (and trials in other conditions).
            if condition not in CONDITIONS:
                continue
            
            # Skip trials where the stimulus rects don't make sense with the
            # viewport, which should result in only one of the two AOI rects
            # being defined (as both will be lower than half the viewport).
            if (aoi_rect["left_stim"] is None) or \
                (aoi_rect["right_stim"] is None):
                print("\t\tAOI locations for participant " + \
                    "'{}' do not line up with viewrect {}".format(\
                    participant, viewrect))
                break
            
            # Skip empty trials.
            if data[participant]["trials"][i]["time"].shape[0] == 0:
                continue
            
            # Add 1 to the stimulus count.
            if stimuli["affective"] not in stim_count.keys():
                stim_count[stimuli["affective"]] = 0
            else:
                stim_count[stimuli["affective"]] += 1
            
            # Skip if we ended up with more repetitions of this stimulus than
            # we should have. (This can happen if participants somehow manage
            # to restart the task; super annoying.)
            if stim_count[stimuli["affective"]] >= NTRIALSPERSTIM:
                continue
            
            # Get the indices for the dwell matrix for the current condition
            # and stimulus.
            i_condition = CONDITIONS.index(condition)
            i_stimulus = STIMULI[condition].index(stimuli["affective"])
            
            # Subtract the trial starting time.
            data[participant]["trials"][i]["time"] -= \
                data[participant]["trials"][i]["time"][0]

            # Go through all samples.
            for sample_i in range(data[participant]["trials"][i]["time"].shape[0]):
                
                # Convenience renaming of the starting time for this sample.
                t0 = data[participant]["trials"][i]["time"][sample_i]
                # Compute the end time of the current sample as the start
                # time of the next.
                if sample_i < data[participant]["trials"][i]["time"].shape[0]-1:
                    t1 = data[participant]["trials"][i]["time"][sample_i+1]
                # If this is the last sample, assume the median inter-sample
                # time.
                else:
                    t1 = t0 + numpy.nanmedian(numpy.diff( \
                        data[participant]["trials"][i]["time"]))
                # Convenience renaming of (x,y) coordinates.
                x = data[participant]["trials"][i]["x"][sample_i]
                y = data[participant]["trials"][i]["y"][sample_i]
                
                # Skip missing data.
                if numpy.isnan(x) or numpy.isnan(y):
                    continue
                # Skip timepoints beyond the edge.
                if t0 > bin_edges[-1]:
                    continue
                # Skip timepoints that are negative. (Somehow this can happen;
                # looks like a task restart halfway through a MouseView zone?)
                if (t0 < 0) or (t1 < 0):
                    continue
                # Skip over-numerous stimuli (this can happen on task restarts)
                if stim_count[stimuli["affective"]] >= NTRIALSPERSTIM:
                    continue
                
                # Store the (x,y) coordinate in the scanpath.
                si = numpy.argmin(numpy.abs(t_samples-t0))
                ei = numpy.argmin(numpy.abs(t_samples-t1))
                scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 0, si:ei] = round(x)
                scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 1, si:ei] = round(y)

                # Check which area of interest this stimulus falls in.
                aoi = "other"
                for aoi_loc in aoi_rect.keys():
                    hor = (x > aoi_rect[aoi_loc][0]) and \
                        (x < aoi_rect[aoi_loc][0]+aoi_rect[aoi_loc][2])
                    ver = (y > aoi_rect[aoi_loc][1]) and \
                        (y < aoi_rect[aoi_loc][1]+aoi_rect[aoi_loc][3])
                    if hor and ver:
                        if aoi_loc == affective_stim:
                            aoi = "affective"
                        else:
                            aoi = "neutral"
                
                # Only record affective and neutral.
                if aoi == "other":
                    continue
                
                # Compute the first bin that this sample falls into. The last
                # bin in the range with smaller edges is the one that the
                # current sample fits within.
                si = numpy.where(bin_edges <= t0)[0][-1]
                
                # Compute the final bin that this sample falls into. The first
                # bin with a larger edge than the current sample will be the
                # bin after the current sample.
                if t1 < bin_edges[-1]:
                    ei = numpy.where(bin_edges > t1)[0][0] - 1
                else:
                    # Minus 2, as we need the last bin, which starts with the
                    # second-to-last bin edge.
                    ei = bin_edges.shape[0] - 2
                
                # If the sample falls within a bin.
                if si == ei:
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), si] \
                        += (t1 - t0) / BINWIDTH
                # If the sample falls in more than one bin.
                else:
                    # Compute the proportion of the first bin that is covered by
                    # the current sample.
                    p0 = (bin_edges[si+1]-t0) / BINWIDTH
                    # Compute the proportion of the last bin that is covered by
                    # the current sample.
                    p1 = (t1 - bin_edges[ei]) / BINWIDTH
                    
                    # Add the proportions to the dwell matrix.
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), si] \
                        += p0
                    dwell[pi, i_condition, i_stimulus, \
                        stim_count[stimuli["affective"]], AOI.index(aoi), ei] \
                        += p1
                    if ei - si > 1:
                        dwell[pi, i_condition, i_stimulus, \
                            stim_count[stimuli["affective"]], AOI.index(aoi), \
                            si+1:ei] = 1.0
            
            # Create a trial plot.
            if CREATE_TRIAL_PLOTS:
                wh_ratio = dispsize[0] / float(dispsize[1])
                fig_w = 8.0
                fig_size = (fig_w, fig_w/wh_ratio)
                fig, ax = pyplot.subplots(figsize=fig_size, dpi=100.0, \
                    nrows=1, ncols=1)
                fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
                # Set limits.
                ax.set_xlim(0, dispsize[0])
                ax.set_ylim(0, dispsize[1])
                # Plot NOT viewrect.
                noview = Rectangle((0,viewrect[1]), viewrect[0], \
                    dispsize[1]-viewrect[1], edgecolor="none", \
                    facecolor="#000000")
                ax.add_patch(noview)
                # Plot viewrect.
                viewport = Rectangle((0,0), viewrect[0], viewrect[1], \
                    edgecolor="#000000", facecolor="none", lw=3, alpha=0.5)
                ax.add_patch(viewport)
                # Plot images.
                img = {}
                for rect_name in aoi_rect.keys():
                    # Draw a rectangle to indicate the stimulus.
                    if rect_name == affective_stim:
                        col = "#c4a000"
                        stim_name = stimuli["affective"]
                    else:
                        col = "#4e9a06"
                        stim_name = stimuli["neutral"]
                    rect = aoi_rect[rect_name]
                    stimrect = Rectangle((rect[0], rect[1]), rect[2], \
                        rect[3], edgecolor=col, facecolor="none", lw=3)
                    ax.add_patch(stimrect)
                    # Annotate the stimulus name.
                    if rect_name == affective_stim:
                        ax.annotate(stimuli["affective"], \
                            (rect[0],rect[1]-rect[3]*0.1), \
                            fontsize=8, color=col)
                    # Plot the image.
                    img[rect_name] = pyplot.imread(os.path.join(STIMDIR, \
                        "{}.jpg".format(stim_name)))
                    ax_img = ax.imshow(img[rect_name], extent=(rect[0], \
                        rect[0]+rect[2], rect[1], rect[1]+rect[3]))
                # Plot scanpath.
                ax.plot( \
                    scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 0, :], \
                    scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 1, :], \
                    ".", color="#FF69B4")
                # Save and close.
                fig.savefig(os.path.join(trial_outdir, \
                    "trial-{}_condition-{}_stimnr-{}_repetition-{}.png".format( \
                    i+1, condition, i_stimulus+1, \
                    stim_count[stimuli["affective"]])))
                pyplot.close(fig)

            # Flip the x-coordinates if the affective stimulus is on the
            # right.
            if affective_stim == "right_stim":
                scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 0, :] = \
                    dispsize[0] - scan[pi, i_condition, i_stimulus, \
                    stim_count[stimuli["affective"]], 0, :]

    # CSV OUTPUT
    # Wide format.
    with open(os.path.join(OUTDIR, "dwell_means_wide.csv"), "w") as f:
        header = ["ppname"]
        for ci, con in enumerate(CONDITIONS):
            for i in range(NTRIALSPERSTIM):
                for j, stim in enumerate(AOI[:2]):
                    header.append("{}_{}_{}".format(con, i+1, stim))
        f.write(",".join(map(str, header)))
        for pi, participant in enumerate(participants):
            line = [participant]
            for ci, con in enumerate(CONDITIONS):
                for i in range(NTRIALSPERSTIM):
                    for j, stim in enumerate(AOI[:2]):
                        # Average over time.
                        d = numpy.nanmean(dwell[pi,ci,:,i,j,:], axis=1)
                        # Average over stimuli.
                        d = numpy.nansum(d, axis=0) / len(STIMULI[con])
                        if d == 0:
                            d = numpy.nan
                        # Add to the line.
                        line.append(d)
            # Write to file.
            f.write("\n" + ",".join(map(str, line)))

    # Long format.
    with open(os.path.join(OUTDIR, "dwell_means_long.csv"), "w") as f:
        if "_" in CONDITIONS[0]:
            header = ["ppname", "group", "environment", "disgust_type", \
                "repetition", "stimulus", "stim_nr", "dwell"]
        else:
            header = ["ppname", "gender", "weaning_status", "group", "condition", \
                "repetition", "stimulus", "stim_nr", "dwell"]
        f.write(",".join(map(str, header)))
        for pi, participant in enumerate(participants):
            line = [participant]
            if "gender" in header:
                psi = numpy.where(parent_status["ppname"] == participant)[0][0]
                gender = parent_status["gender"][psi]
            for ci, con in enumerate(CONDITIONS):
                if "_" in con:
                    environment, disgust_type = con.split("_")
                else:
                    environment = None
                    disgust_type = None
                for i in range(NTRIALSPERSTIM):
                    for j, stim in enumerate(AOI[:2]):
                        for stim_nr in range(len(STIMULI[con])):
                            # Average over time bins.
                            d = numpy.nanmean(dwell[pi,ci,stim_nr,i,j,:])
                            if "_" in con:
                                line = [participant, ppgroups[pi], environment, \
                                    disgust_type, i, stim, stim_nr, d]
                            else:
                                line = [participant, gender, weaning_status[pi], \
                                    ppgroups[pi], con, i, stim, stim_nr, d]
                            # Write to file.
                            f.write("\n" + ",".join(map(str, line)))

# Load the data from the temporary file.
else:
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

    # Load the scanpath data.
    scan_shape = tuple(numpy.memmap(SCANSHAPEPATH, dtype=numpy.int32, \
        mode="r"))
    scan = numpy.memmap(SCANPATH, dtype=numpy.int16, mode="r", \
        shape=scan_shape)
    # Recompute the time for each sample.
    n_samples = int(numpy.ceil((TRIAL_DURATION / 1000.0) * SAMPLE_RATE))
    t_samples = numpy.linspace(0, (1000.0/SAMPLE_RATE)*(n_samples-1), \
        n_samples)


# # # # #
# PLOT

# The dwell matrix has shape (n_participants, n_conditions, 
# n_stimuli_per_condition, n_stimulus_presentations, n_aois, n_bins)
# Plotting should happen as a time series (dwell[:,:,:,:,:,0:t). We're 
# interested in the difference between the neutral AOI  (dwell[:,:,:,:,1,:])
#  and the affective AOI (dwell[:,:,:,:,0,:]). This should be averaged across
# all stimuli (dwell[:,:,0:n,:,:,:]), and plotted separately for each stimulus
# presentation (dwell[:,:,:,0:NTRIALSPERSTIM,:,:]). The conditions 
# (dwell[:,i,:,:,:,:]) should be plotted in separate plots. The average dwell
# time difference and 95% confidence interval should be computed over all
# participants (dwell[0:n_participants,:,:,:,:,:]).

# Create a four-panel figure. (Top row for lines, bottom row for heatmaps;
# columns for different conditions.)
fig, axes = pyplot.subplots(nrows=2, ncols=len(CONDITIONS), \
    figsize=(8.0*len(CONDITIONS),9.0), dpi=300.0)
fig.subplots_adjust(left=0.05, bottom=0.06, right=0.95, top=0.95,
    wspace=0.15, hspace=0.2)

# Compute the bin centres to serve as x-values.
time_bin_centres = bin_edges[:-1] + numpy.diff(bin_edges)/2.0

# Loop through the conditions.
for ci, condition in enumerate(CONDITIONS):
    
    # Choose the top-row, and the column for this condition.
    ax = axes[0,ci]
    
    # Compute the difference between neutral and affective stimulus. This
    # quantifies approach (positive values) or avoidance (negative values).
    # Computed as affective minus neutral, only for the current condition.
    # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
    d = dwell[:,ci,:,:,0,:] - dwell[:,ci,:,:,1,:]
    
    # Average over all different stimuli, and recode as percentages.
    # val has shape (n_participants, n_presentations, n_bins)
    val = 100 * numpy.nansum(d, axis=1) / len(STIMULI[condition])
    
    # Compute the mean over all participants.
    m = numpy.nanmean(val, axis=0)
    # Compute the within-participant 95% confidence interval.
    nv = val - numpy.nanmean(val, axis=0) \
        + numpy.nanmean(numpy.nanmean(val, axis=0))
    sd = numpy.nanstd(nv, axis=0, ddof=1)
    sem = sd / numpy.sqrt(nv.shape[0])
    ci_95 = 1.96 * sem

    # Specify the colour map for the current condition.
    cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
    voffset = 3
    vmin = 0
    vmax = dwell.shape[3] + voffset
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Plot the separation line and background colours.
    ax.plot(time_bin_centres, numpy.zeros(time_bin_centres.shape), ':', lw=3, \
        color="black", alpha=0.5)
    ax.fill_between(time_bin_centres, \
        numpy.ones(time_bin_centres.shape)*YLIM["dwell_p"][0], \
        numpy.zeros(time_bin_centres.shape), color="black", alpha=0.05)
    annotate_x = time_bin_centres[0] + \
        0.02*(time_bin_centres[-1]-time_bin_centres[0])
    ax.annotate("Approach", (annotate_x, YLIM["dwell_p"][1]-5), \
        fontsize=FONTSIZE["annotation"])
    ax.annotate("Avoidance", (annotate_x, YLIM["dwell_p"][0]+3), \
        fontsize=FONTSIZE["annotation"])
    
    # LINES
    # Plot each stimulus presentation separately.
    for j in range(m.shape[0]):
        
        # Define the label.
        if j == 0:
            lbl = "First presentation"
        else:
            lbl = "Stimulus repeat {}".format(j)

        # PLOT THE LIIIIIIINE!
        ax.plot(time_bin_centres, m[j,:], "-", color=cmap(norm(j+voffset)), \
            lw=3, label=lbl)
        # Shade the confidence interval.
        ax.fill_between(time_bin_centres, m[j,:]-ci_95[j,:], \
            m[j,:]+ci_95[j,:], alpha=0.2, color=cmap(norm(j+voffset)))
        
    # Add a legend.
    if condition in LEGENDLOCS.keys():
        loc = LEGENDLOCS[condition]
    else:
        loc = "best"
    ax.legend(loc=loc, fontsize=FONTSIZE["legend"])

    # Finish the upper plot.
    ax.set_title(condition.capitalize(), fontsize=FONTSIZE["axtitle"])
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([0, TRIAL_DURATION])
    ax.set_xticks(range(0, TRIAL_DURATION+1, 1000))
    ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
        fontsize=FONTSIZE["ticklabels"])
    if ci == 0:
        ax.set_ylabel(r"Dwell percentage difference", fontsize=FONTSIZE["label"])
    ax.set_ylim(YLIM["dwell_p"])
    ax.set_yticks(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10))
    ax.set_yticklabels(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10), \
        fontsize=FONTSIZE["ticklabels"])
    
    # HEATMAPS
    # Choose the heatmap row, and the column for the current condition.
    ax = axes[1,ci]

    # Create the colourmap for this condition.
    cmap = matplotlib.cm.get_cmap("coolwarm")
    vmax = max(abs(YLIM["dwell_p"][0]), abs(YLIM["dwell_p"][1]))
    vmin = -1 * vmax
    vstep = YLIM["dwell_p"][1] // 2

    # Plot the heatmap.
    ax.imshow(m, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none", \
        aspect="auto", origin="upper")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(ax)
    bax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = matplotlib.colorbar.ColorbarBase(bax, cmap=cmap, norm=norm, \
        ticks=range(vmin, vmax+1, vstep), orientation='vertical')
    if ci == axes.shape[1]-1:
        cbar.set_label(r"$\Delta$dwell percentage", \
            fontsize=FONTSIZE["bar"])

    #ax.set_title(CONDITIONS[condition], fontsize=FONTSIZE["axtitle"])
    ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
    ax.set_xlim([-0.5, m.shape[1]-0.5])
    ax.set_xticks(numpy.linspace(-0.5, m.shape[1]+0.5, \
        num=TRIAL_DURATION//1000 + 1))
    ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
        fontsize=FONTSIZE["ticklabels"])
    if ci == 0:
        ax.set_ylabel("Presentation number", fontsize=FONTSIZE["label"])
    ax.set_ylim([NTRIALSPERSTIM-0.5, -0.5])
    ax.set_yticks(range(0, NTRIALSPERSTIM))
    ax.set_yticklabels(range(1, NTRIALSPERSTIM+1), \
        fontsize=FONTSIZE["ticklabels"])

# Save the figure.
fig.savefig(os.path.join(OUTDIR, "dwell_percentages.png"))
pyplot.close(fig)


# # # # #
# PLOT

# The dwell matrix has shape (n_participants, n_conditions, 
# n_stimuli_per_condition, n_stimulus_presentations, n_aois, n_bins)
# Plotting should happen as a time series (dwell[:,:,:,:,:,0:t). We're 
# interested in the difference between the neutral AOI  (dwell[:,:,:,:,1,:])
#  and the affective AOI (dwell[:,:,:,:,0,:]). This should be averaged across
# all stimuli (dwell[:,:,0:n,:,:,:]), and plotted separately for each stimulus
# presentation (dwell[:,:,:,0:NTRIALSPERSTIM,:,:]). The conditions 
# (dwell[:,i,:,:,:,:]) should be plotted in separate plots. The average dwell
# time difference and 95% confidence interval should be computed over all
# participants (dwell[0:n_participants,:,:,:,:,:]).

for grouping_type in ["group", "weaning_status"]:
    
    if grouping_type == "group":
        group_names = numpy.unique(ppgroups)
    elif grouping_type == "weaning_status":
        group_names = numpy.unique(weaning_status)
    n_rows = group_names.shape[0]
    n_cols = 4
    w = n_cols * 6.0 #8.27
    h = n_rows * ((w / n_cols) / 1.6)
    fig, axes = pyplot.subplots(nrows=n_rows, ncols=n_cols, \
        sharex=True, sharey=True, figsize=(w,h), dpi=300.0)
    fig.subplots_adjust(left=0.07, bottom=0.06, right=0.98, top=0.95,
        wspace=0.08, hspace=0.1)
    
    for gi, group_name in enumerate(group_names):
        
        # Compute the bin centres to serve as x-values.
        time_bin_centres = bin_edges[:-1] + numpy.diff(bin_edges)/2.0
        
        # Loop through the conditions.
        for ci, condition in enumerate(CONDITIONS):
            
            # Choose the ax to draw in.
            row = gi
            col = ci
            ax = axes[row,col]

            # Create a Boolean mask to only include participants in the current
            # group.
            if grouping_type == "group":
                pp_sel = ppgroups == group_name
            elif grouping_type == "weaning_status":
                pp_sel = weaning_status == group_name

            # Compute the difference between neutral and affective stimulus. This
            # quantifies approach (positive values) or avoidance (negative values).
            # Computed as affective minus neutral, only for the current condition.
            # d has shape (n_participants, n_stimuli, n_presentations, n_bins)
            d = dwell[pp_sel,ci,:,:,0,:] - dwell[pp_sel,ci,:,:,1,:]
            
            # Average over all different stimuli, and recode as percentages.
            # val has shape (n_participants, n_presentations, n_bins)
            val = 100 * numpy.nansum(d, axis=1) / len(STIMULI[condition])
            
            # Compute the mean over all participants.
            m = numpy.nanmean(val, axis=0)
            # Compute the within-participant 95% confidence interval.
            nv = val - numpy.nanmean(val, axis=0) \
                + numpy.nanmean(numpy.nanmean(val, axis=0))
            sd = numpy.nanstd(nv, axis=0, ddof=1)
            sem = sd / numpy.sqrt(nv.shape[0])
            ci_95 = 1.96 * sem
        
            # Specify the colour map for the current condition.
            cmap = matplotlib.cm.get_cmap(PLOTCOLMAPS[condition])
            voffset = 3
            vmin = 0
            vmax = dwell.shape[3] + voffset
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            
            # Plot the separation line and background colours.
            ax.plot(time_bin_centres, numpy.zeros(time_bin_centres.shape), ':', lw=3, \
                color="black", alpha=0.5)
            ax.fill_between(time_bin_centres, \
                numpy.ones(time_bin_centres.shape)*YLIM["dwell_p"][0], \
                numpy.zeros(time_bin_centres.shape), color="black", alpha=0.1)
            annotate_x = time_bin_centres[0] + \
                0.02*(time_bin_centres[-1]-time_bin_centres[0])
            ax.annotate("Approach", (annotate_x, YLIM["dwell_p"][1]-5), \
                fontsize=FONTSIZE["annotation"])
            ax.annotate("Avoidance", (annotate_x, YLIM["dwell_p"][0]+3), \
                fontsize=FONTSIZE["annotation"])
            
            # LINES
            # Plot each stimulus presentation separately.
            for j in range(m.shape[0]):
                
                # Define the label.
                if j == 0:
                    lbl = "First presentation"
                else:
                    lbl = "Stimulus repeat {}".format(j)
        
                # PLOT THE LIIIIIIINE!
                ax.plot(time_bin_centres, m[j,:], "-", color=cmap(norm(j+voffset)), \
                    lw=3, label=lbl)
                # Shade the confidence interval.
                ax.fill_between(time_bin_centres, m[j,:]-ci_95[j,:], \
                    m[j,:]+ci_95[j,:], alpha=0.2, color=cmap(norm(j+voffset)))
                
            # Add a legend.
            if condition in LEGENDLOCS.keys():
                loc = LEGENDLOCS[condition]
            else:
                loc = "best"
            if row == n_rows-1:
                ax.legend(loc=loc, fontsize=FONTSIZE["legend"])
        
            # Finish plot.
            if row == 0:
                if condition == "control":
                    title = "Bodily effluvia"
                else:
                    title = "{} diaper".format(condition.capitalize())
                ax.set_title(title, fontsize=FONTSIZE["axtitle"])
            if row == n_rows-1:
                ax.set_xlabel("Time in trial (seconds)", fontsize=FONTSIZE["label"])
            ax.set_xlim([0, TRIAL_DURATION])
            ax.set_xticks(range(0, TRIAL_DURATION+1, 1000))
            ax.set_xticklabels(range(0, (TRIAL_DURATION//1000 + 1)), \
                fontsize=FONTSIZE["ticklabels"])
            if col == 0:
                ax.set_ylabel("{} group (N={})\n\nDwell difference (% pt.)".format( \
                    group_name.capitalize(), numpy.sum(pp_sel)), \
                    fontsize=FONTSIZE["label"])
            ax.set_ylim(YLIM["dwell_p"])
            ax.set_yticks(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10))
            ax.set_yticklabels(range(YLIM["dwell_p"][0], YLIM["dwell_p"][1]+1, 10), \
                fontsize=FONTSIZE["ticklabels"])
        
    # Save the figure.
    fig.savefig(os.path.join(OUTDIR, "disgust_avoidance_{}.png".format( \
        grouping_type)))
    pyplot.close(fig)
    
