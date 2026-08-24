#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript analysis/channel2/prepare-experiment.R RUN_ID OUTPUT_DIR", call. = FALSE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
here <- dirname(sub("^--file=", "", script_arg))
source(file.path(here, "experiment-inputs.R"), local = TRUE)
experiment_prepare_inputs(args[[1L]], normalizePath(args[[2L]], mustWork = FALSE))
