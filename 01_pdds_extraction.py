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
FILENAME = "data_exp_163666-v4_questionnaire-kdzi.csv"

# QUESTIONNAIRE
# Number of PDDS questions.
N_QUESTIONS = 13
# Construct the questions we need to extract data for.
QUESTIONS = ["pdds{}".format(i+1) for i in range(N_QUESTIONS)]
# Due to a WHOOPSIE, "pdds4" is actually "tdds4".
qi = QUESTIONS.index("pdds4")
QUESTIONS[qi] = "tdds4"
# SUBSCALES
PDDS = { \
    "parent":   list(range(1,N_QUESTIONS+1)), \
    }

# EXCLUSIONS
# Excluded participants.
EXCLUDED = [ \
    ]

# FILES AND FOLDERS.
DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(DIR, "data", DATAFOLDER)
FILEPATH = os.path.join(DATADIR, FILENAME)
OUTDIR = os.path.join(DIR, "output_{}".format(DATAFOLDER))
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# LOAD DATA.

# Start with an empty dictionary.
data = {}

# Loop through all lines of data.
with open(os.path.join(DATADIR, FILENAME), "r") as f:

    # Start without any info on the header and the indices of relevant columns.
    header = None
    pi = None
    qi = None
    ri = None

    # Loop through all lines in this file.
    for i, line in enumerate(f):
        
        # Remove the trailing newline.
        line = line.replace("\n", "")

        # Stop if we reached the end of the file.
        if line == "END OF FILE":
            break

        # Split by commas.
        line = line.split(",")

        # The first line is the header.
        if header is None:
            # Copy the header.
            header = line[:]
            # Extract the indices for data we need.
            pi = header.index("Participant Public ID")
            qi = header.index("Question Key")
            ri = header.index("Response")
            # Skip processing, as this is not a data line.
            continue
        
        # Only process lines with data for the questions we need.
        if line[qi] not in QUESTIONS:
            continue
        
        # Check if this is a new participant.
        if line[pi] not in data.keys():
            data[line[pi]] = numpy.zeros(N_QUESTIONS, dtype=numpy.float64) \
                * numpy.NaN
        
        # Store the response.
        i = QUESTIONS.index(line[qi])
        data[line[pi]][i] = int(line[ri])


# # # # #
# WRITE TO FILE

with open(os.path.join(OUTDIR, "questionnaire_pdds.csv"), "w") as f:
    
    # Construct a header.
    header = ["ppname"]
    for qname in QUESTIONS:
        # Revert the WHOOPSIE.
        if qname == "tdds4":
            qname = "pdds4"
        # Get the number from the question name.
        qnr = int(qname.replace("pdds", ""))
        # Find which subscale this question was in.
        for subscale in PDDS.keys():
            if qnr in PDDS[subscale]:
                break
        # Construct the full question name with number and subscale.
        header.append("pdds_{}_{}".format(qnr, subscale))

    # Write the header to the file.
    f.write(",".join(header))
    
    # Loop through all data.
    for ppname in data.keys():
        line = [ppname] + list(data[ppname].astype(numpy.int64))
        f.write("\n" + ",".join(map(str,line)))
    
