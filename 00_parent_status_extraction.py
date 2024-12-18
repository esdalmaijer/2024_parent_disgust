#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
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
FILENAME = "data_exp_163666-v4_questionnaire-ytcf.csv"

# QUESTIONNAIRE
# Construct the questions we need to extract data for.
QUESTIONS = [ \
    "number_of_children", \
    "age_months_only_child", \
    "age_months_youngest", \
    "age_year_oldest", \
    "nappy_change-frequency", \
    "weaning_youngest", \
    "weaning_any", \
    "nappy_change_wee", \
    "nappy_change_poo", \
    "gender", \
    "participant_age_years", \
    ]
# Number of questions we're extracting for each participant.
N_QUESTIONS = len(QUESTIONS)

# EXCLUSIONS
# Excluded participants.
EXCLUDED = [ \
    ]

# Specifics of questionnaire.
GENDER_OPTIONS = ["Prefer not to say", "Man", "Woman", "Non-binary"]
ETHNICITY_DICT = { \
    "ethnicity_no-resp": """Prefer not to say""", \
    "ethnicity_indigenous": """American Indian, Alaskan Native, Aboriginal Australians, Maori – for example, Navajo Nation, Blackfeet Tribe, Mayan, Aztec, Native Village of Barrow Inupiat Traditional Government, Nome Eskimo Community, Anindilyakwa, Arrernte, Bininj, Gunggari, Muruwari""", \
    "ethnicity_asian": """Asian – for example, Chinese, Filipino, Asian Indian, Vietnamese, Korean, Japanese""", \
    "ethnicity_black": """Black or – for example, Jamaican, Haitian, Nigerian, Ethiopian, Somalian""", \
    "ethnicity_latine": """Latino/Latina/Latine – for example, Mexican, Puerto Rican, Cuban, Salvadoran, Dominican, Columbian""", \
    "ethnicity_pacific": """Pacific Islanders – for example, Native Hawaiian, Samoan, Chamorro, Tongan, Fijian, Marshallese""", \
    "ethnicity_white": """White – for example, German, Irish, English, Scottish, Welsh, Northern Irish, British, Traveller, Italian, French""", \
    }
ETHNICITY_OPTIONS = list(ETHNICITY_DICT.keys())
ETHNICITY_OPTIONS.sort()

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
with open(os.path.join(DATADIR, FILENAME), mode="r", encoding="utf-8") as f:

    # Snif out the dialect. We're using a large chunk of data for this,
    # as the header gets quite big.
    dialect = csv.Sniffer().sniff(f.read())
    # Check the delimiter.
    print("Auto-detected delimiter: '{}'".format( \
        dialect.delimiter))
    
    # Enforce a few things in the dialect.
    # dialect.quoting = 1
    dialect.quotechar = '"'
    dialect.doublequote = True
    dialect.skipinitialspace = True

    # Set the reading position back to the start of the file.
    f.seek(0)
    # Start a CSV reader.
    reader = csv.reader(f, dialect)
    # reader = csv.reader(f, skipinitialspace=True)
    
    # Start without any info on the header and the indices of relevant columns.
    header = None
    pi = None
    qi = None
    ri = None

    # Loop through all lines in this file.
    while True:
        
        line = next(reader)
        
        # Stop if we reached the end of the file.
        if "END OF FILE" in line:
            break

        # The first line is the header.
        if header is None:
            # Copy the header.
            header = line[:]
            # Extract the indices for data we need.
            pi = header.index("Participant Public ID")
            qi = header.index("Object Name")
            ki = header.index("Key")
            ri = header.index("Response")
            oi = header.index("OptionOrder")
            # Skip processing, as this is not a data line.
            continue
        
        # Only process lines with data for the questions we need.
        is_question_row = (line[qi] in QUESTIONS) and (line[ki] == "value")
        is_ethnicity_row = line[qi] == "ethnicity"
        if not (is_question_row or is_ethnicity_row):
            continue
        
        # Check if this is a new participant.
        if line[pi] not in data.keys():
            data[line[pi]] = numpy.zeros(N_QUESTIONS + len(ETHNICITY_OPTIONS), \
                dtype=numpy.float64) * numpy.NaN
        
        # Skip if there is no response.
        if line[ri] == "":
            continue

        # Store the ethnicity response.
        if is_ethnicity_row:
            for i, option in enumerate(ETHNICITY_OPTIONS):
                answer = -1
                if line[ki] == ETHNICITY_DICT[option]:
                    answer = int(line[ri])
                    i += N_QUESTIONS
                    data[line[pi]][i] = answer
                    break
            continue

        # Store the response.
        if line[qi] in ["nappy_change-frequency", "weaning_youngest", \
            "weaning_any", "number_of_children", "gender"]:
            options = line[oi].split("|")
            answer = options.index(line[ri])
            if line[qi] == "weaning_any":
                answer = 1 - answer
        else:
            try:
                answer = int(line[ri])
            except ValueError:
                answer = round(float(line[ri]))
        i = QUESTIONS.index(line[qi])
        data[line[pi]][i] = answer


# # # # #
# WRITE TO FILE

with open(os.path.join(OUTDIR, "questionnaire_parent_status.csv"), "w") as f:
    
    # Construct a header.
    header = ["ppname"] + QUESTIONS + ETHNICITY_OPTIONS
    
    # Write the header to the file.
    f.write(",".join(header))
    
    # Loop through all data.
    for ppname in data.keys():
        line = [ppname] + list(data[ppname])
        line[1+QUESTIONS.index("gender")] = \
            GENDER_OPTIONS[int(line[1+QUESTIONS.index("gender")])]
        f.write("\n" + ",".join(map(str,line)))
    
