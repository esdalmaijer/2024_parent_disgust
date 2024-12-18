#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy
import pandas
import scipy.stats
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.power import tt_solve_power
from statsmodels.tools.eval_measures import aic, bic
from sklearn.impute import KNNImputer


# # # # #
# SETUP

DATAFOLDER = "data_exp_163666-v4_2024-08-21"

# Name of the group to be focussed on, or "combined" to use all groups.
GROUP = "combined"

# Define models.
FORMULA = { \
    "combined": { \
        "dwell": [ \
            "dwell ~ group + stimulus + condition + stim_nr + repetition + " \
                + "group*stimulus*condition*stim_nr*repetition", \
            "dwell ~ group + stimulus + condition + stim_nr + " \
                + "group*stimulus*condition*stim_nr", \
            "dwell ~ group + stimulus + condition + repetition + " \
                + "group*stimulus*condition*repetition", \
            "dwell ~ group + stimulus + stim_nr + repetition + " \
                + "group*stimulus*stim_nr*repetition", \
            "dwell ~ group + stimulus + condition + " \
                + "group*stimulus*condition", \
            "dwell ~ group + stimulus + stim_nr + " \
                + "group*stimulus*stim_nr", \
            "dwell ~ gender + group + stimulus + repetition + " \
                + "gender*group*stimulus*repetition", \
            "dwell ~ group + stimulus + repetition + " \
                + "group*stimulus*repetition", \
            "dwell ~ group + stimulus + " \
                + "group*stimulus", \
            "dwell ~ group", \

            "dwell ~ weaning_status + stimulus + condition + stim_nr + repetition + " \
                + "weaning_status*stimulus*condition*stim_nr*repetition", \
            "dwell ~ weaning_status + stimulus + condition + stim_nr + " \
                + "weaning_status*stimulus*condition*stim_nr", \
            "dwell ~ weaning_status + stimulus + condition + repetition + " \
                + "weaning_status*stimulus*condition*repetition", \
            "dwell ~ weaning_status + stimulus + stim_nr + repetition + " \
                + "weaning_status*stimulus*stim_nr*repetition", \
            "dwell ~ weaning_status + stimulus + condition + " \
                + "weaning_status*stimulus*condition", \
            "dwell ~ weaning_status + stimulus + stim_nr + " \
                + "weaning_status*stimulus*stim_nr", \
            "dwell ~ weaning_status + stimulus + repetition + " \
                + "weaning_status*stimulus*repetition", \
            "dwell ~ weaning_status + stimulus + " \
                + "weaning_status*stimulus", \
            "dwell ~ weaning_status", \

            "dwell ~ stimulus + condition + stim_nr + repetition + " \
                + "stimulus*condition*stim_nr*repetition", \
            "dwell ~ stimulus + condition + stim_nr + " \
                + "stimulus*condition*stim_nr", \
            "dwell ~ stimulus + condition + repetition + " \
                + "stimulus*condition*repetition", \
            "dwell ~ stimulus + stim_nr + repetition + " \
                + "stimulus*stim_nr*repetition", \
            "dwell ~ stimulus + condition + " \
                + "stimulus*condition", \
            "dwell ~ stimulus + stim_nr + " \
                + "stimulus*stim_nr", \
            "dwell ~ stimulus + repetition + " \
                + "stimulus*repetition", \
            "dwell ~ stimulus", \
            "dwell ~ 1", \
                ], \
        }, \

    # The following can be used for group-specific analyses.
    "milk": { \
        "dwell": [ \
            "dwell ~ stimulus + condition + stim_nr + repetition + " \
                + "stimulus*condition*stim_nr*repetition", \
            "dwell ~ stimulus + condition + stim_nr + " \
                + "stimulus*condition*stim_nr", \
            "dwell ~ stimulus + condition + repetition + " \
                + "stimulus*condition*repetition", \
            "dwell ~ stimulus + stim_nr + repetition + " \
                + "stimulus*stim_nr*repetition", \
            "dwell ~ stimulus + condition + " \
                + "stimulus*condition", \
            "dwell ~ stimulus + stim_nr + " \
                + "stimulus*stim_nr", \
            "dwell ~ stimulus + repetition + " \
                + "stimulus*repetition", \
            "dwell ~ stimulus" \
                ], \
        }, \

    "weaned": { \
        "dwell": [ \
            "dwell ~ stimulus + condition + stim_nr + repetition + " \
                + "stimulus*condition*stim_nr*repetition", \
            "dwell ~ stimulus + condition + stim_nr + " \
                + "stimulus*condition*stim_nr", \
            "dwell ~ stimulus + condition + repetition + " \
                + "stimulus*condition*repetition", \
            "dwell ~ stimulus + stim_nr + repetition + " \
                + "stimulus*stim_nr*repetition", \
            "dwell ~ stimulus + condition + " \
                + "stimulus*condition", \
            "dwell ~ stimulus + stim_nr + " \
                + "stimulus*stim_nr", \
            "dwell ~ stimulus + repetition + " \
                + "stimulus*repetition", \
            "dwell ~ stimulus" \
                ], \
        }, \

    "control": { \
        "dwell": [ \
            "dwell ~ stimulus + condition + stim_nr + repetition + " \
                + "stimulus*condition*stim_nr*repetition", \
            "dwell ~ stimulus + condition + stim_nr + " \
                + "stimulus*condition*stim_nr", \
            "dwell ~ stimulus + condition + repetition + " \
                + "stimulus*condition*repetition", \
            "dwell ~ stimulus + stim_nr + repetition + " \
                + "stimulus*stim_nr*repetition", \
            "dwell ~ stimulus + condition + " \
                + "stimulus*condition", \
            "dwell ~ stimulus + stim_nr + " \
                + "stimulus*stim_nr", \
            "dwell ~ stimulus + repetition + " \
                + "stimulus*repetition", \
            "dwell ~ stimulus" \
                ], \
        }, \
    }

# Set the variables that need to be standardised. (all non-categorical ones.)
STANDARDISE_THESE = ["dwell"]
# Impute missing variables?
IMPUTE_MISSING = False

# FILES AND FOLDERS
# Path the this file's directory.
DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data directory.
DATADIR = os.path.join(DIR, "output_{}".format(DATAFOLDER))
if not os.path.isdir(DATADIR):
    raise Exception("ERROR: Could not find data directory at '%s'" % (DATADIR))
OUTDIR = os.path.join(DIR, "output_{}".format(DATAFOLDER))
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)
OUTDIR = os.path.join(OUTDIR, "lme_output")
if IMPUTE_MISSING:
    OUTDIR += "_imputed"
if not os.path.isdir(OUTDIR):
    os.mkdir(OUTDIR)


# # # # #
# IMPUTATION

# Function to go from long to wide and back again, and impute while in wide
# format.
def impute_and_exclude(X, value, index, within_columns, save_as, \
    exclude_prop=0.3, overwrite=True):
    
    X_ = X.copy()
    if (not os.path.isfile(save_as)) or overwrite:
        # Pivot X into wide format.
        Xw_df = X_.pivot_table(values=value, index=index, \
            columns=within_columns, fill_value = numpy.NaN, dropna=False)
        # Grab the header, which is in arrays of unique combinations of 
        # possible values in the comments
        X_wide_header = list(Xw_df.columns.values)
        # Grab the actual values.
        X_wide = Xw_df.values
        
        # Exclude participants with high proportions of missing data.
        if exclude_prop is not None:
            # Compute the proportion of missing values per participant.
            n_missing = numpy.sum(numpy.isnan(X_wide).astype(int), axis=1)
            p_missing = n_missing.astype(float) / X_wide.shape[1]
            # Exclude participants with too many missing values.
            sel = p_missing>exclude_prop
            excluded = list(Xw_df.index[sel])
            excluded_long = numpy.zeros(X_.shape[0], dtype=bool)
            for excluded_id in excluded:
                excluded_long[X_[index] == excluded_id] = True
            X_ = X_.drop(X_.index[excluded_long])
        
        # Perform knn imputation.
        knn = KNNImputer(missing_values=numpy.NaN, n_neighbors=5)
        X_wide = knn.fit_transform(X_wide)
        
        # Pandas consistently fucks up the slicing, so we'll do it as NumPy
        # arrays.
        X_long_header = list(X_.columns.values)
        X_long = X_.values
        # Go through all long-format lines with a missing value.
        for i in numpy.where(numpy.isnan(X_long[:,X_long_header.index(value)].astype(float)))[0]:
            # Find the imputed data's row in the wide column.
            row = numpy.where(Xw_df.index == X_long[i,X_long_header.index(index)])[0][0]
            # Find the imputed value's column in the wide data.
            header_label = []
            for varname in within_columns:
                header_label.append(X_long[i,X_long_header.index(varname)])
            col = X_wide_header.index(tuple(header_label))
            # Place the imputed data in X.
            X_long[i,X_long_header.index(value)] = X_wide[row,col]
        
        # Write to file.
        with open(save_as, "w") as f:
            f.write(",".join(map(str, X_long_header)))
            for i in range(X_long.shape[0]):
                line = list(X_long[i,:])
                f.write("\n" + ",".join(map(str, line)))
        
    # Load and return the data
    return pandas.read_csv(save_as)


# # # # #
# MODELS

# Loop through all data types (dwell and rating)
for datatype in FORMULA[GROUP].keys():

    # Select data to load.
    if datatype == "dwell":
        file_names = ["dwell_means_long"]

    # Loop through all files.
    for fname in file_names:
        
        print("Loading data from {}".format(fname))

        # Load data.
        file_path = os.path.join(DATADIR, fname+".csv")
        data = pandas.read_csv(file_path)
        # If we're not looking at the combined sample, select only those who
        # are in the subgroup that we're testing.
        if GROUP != "combined":
            data = data.drop(data[data["group"] != GROUP].index)
        n_original = len(numpy.unique(data["ppname"]))
        n_excluded = None
        # Gender categories can contain few observations. This is an issue if
        # only one such observation occurs. In this case, we adjust to "man" 
        # and "not_man". The idea behind this is that, while they are different
        # challenges, anyone who identifies as anything other than man is 
        # likelier to face more challenges than men typically do.
        gender_categories = numpy.unique(data["gender"])
        gender_category_count = {}
        any_singular = False
        for category in gender_categories:
            gender_category_count[category] = len(numpy.unique( \
                data["ppname"][data["gender"]==category]))
            if gender_category_count[category] == 1:
                any_singular = True
        if any_singular:
            men = numpy.isin(data["gender"], ["Man", "man"])
            not_men = numpy.invert(men)
            data["gender"][men] = "man"
            data["gender"][not_men] = "not_man"
        
        # Impute missing data.
        if IMPUTE_MISSING:
            print("\tImputing missing data.")
            if datatype == "dwell":
                value = "dwell"
                within_columns = ["condition", "repetition", "stimulus", \
                    "stim_nr"]
                max_missing_prop = 0.3
            save_as = os.path.join(DATADIR, fname+"_imputed.csv")
            data = impute_and_exclude(data, value, "ppname", within_columns, \
                save_as, exclude_prop=max_missing_prop, overwrite=True)
            n_included = len(numpy.unique(data["ppname"]))
            n_excluded = n_original - n_included

        # Standardise the variables that need to be standardised.
        print("\tStandardising data.")
        for var in STANDARDISE_THESE:
            if var in data.keys():
                m = numpy.nanmean(data[var])
                sd = numpy.nanstd(data[var])
                data[var] = (data[var] - m) / sd

        # Open a new output file.
        fpath = os.path.join(OUTDIR, "lme_{}.txt".format(fname))
        with open(fpath, "w") as f:
            # Loop through all specified models.
            for i, formula in enumerate(FORMULA[GROUP][datatype]):
                print("\tFitting model: {}".format(formula))
                # Fit the current model.
                lme = MixedLM.from_formula(formula, groups=data["ppname"], \
                    data=data, missing="drop")
                lme = lme.fit()
                # Write outcomes to file.
                f.write(lme.summary().as_text())
                f.write(formula)
                f.write("\nAIC = {}; BIC = {}".format( \
                    aic(lme.llf, lme.nobs, lme.df_modelwc), \
                    bic(lme.llf, lme.nobs, lme.df_modelwc)))
                if n_excluded is not None:
                    f.write("\nExcluded: {}".format(n_excluded))
                f.write("\n\n\n")
                
                # Write the outcomes for just this model to a separate file.
                f_path_ = os.path.join(OUTDIR, "lme_{}_{}.tsv".format(fname,i))
                with open(f_path_, "w") as f_:
                    # Write header to file.
                    header = ["param", "beta", "se", "t", "p", "95_ci_low", \
                        "95_ci_high", "d", "TOST_t", "TOST_p", \
                        "p(data>low)", "p(data<high)"]
                    f_.write("\t".join(map(str, header)))
                    # Count the number of participants in this analysis.
                    n = len(numpy.unique(data["ppname"]))
                    # Quick power computation to find the smallest effect size.
                    smallest_d = tt_solve_power(nobs=n, alpha=0.5, power=0.8)
                    # Power works differently for equivalence testing; the
                    # limit for alpha=0.5 and power=80% at N=96 is 0.3.
                    tost_d = 0.3
                    # Compute t values, p values, Cohen's d estimates, 
                    # confidence intervals, and equivalence test.
                    # Get the beta estimates.
                    beta = lme.params
                    # Get the parameter names in the beta dict.
                    beta_names = beta.keys()
                    # Compute confidence intervals.
                    ci_95 = lme.conf_int(alpha=0.05)
                    # Grab the standard errors.
                    se = lme.bse
                    # Estimate SD from the standard error.
                    sd = se * numpy.sqrt(n)
                    # Compute t values.
                    t = beta / se
                    # Compute effect size Cohen's d.
                    d = t / numpy.sqrt(n)
                    
                    # Write all parameters to file.
                    for param in beta_names:
                        
                        # Skip the group variance param.
                        if param == "Group Var":
                            continue
                        
                        # Compute a p value for the t test.
                        p = 2.0 * (1.0 - scipy.stats.t.cdf(numpy.abs( \
                            t[param]), n-1))
                        
                        # Equivalence test (TOST)
                        # Test if parameter is larger than low bound.
                        low_bound = -tost_d * sd[param]
                        low_t = (beta[param] - low_bound) / se[param]
                        low_p = 1.0 - scipy.stats.t.cdf(low_t, n-1)
                        # Test if parameter is smaller than high bound.
                        high_bound = tost_d * sd[param]
                        high_t = (beta[param] - high_bound) / se[param]
                        high_p = scipy.stats.t.cdf(high_t, n-1)
                        # TOST t value is the lowest absolute value.
                        if numpy.abs(low_t) < numpy.abs(high_t):
                            tost_t = low_t
                        else:
                            tost_t = high_t
                        # TOST p value is the highest p value.
                        tost_p = numpy.max([low_p, high_p])
                        
                        # Write to file.
                        line = [param, beta[param], se[param], t[param], p, \
                            ci_95[0][param], ci_95[1][param], d[param], \
                            tost_t, tost_p, low_p, high_p]
                        f_.write("\n" + "\t".join(map(str, line)))

