#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Usage: Rscript analysis/channel2/verify-experiment.R RUN_ID RUN_DIR", call. = FALSE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
here <- dirname(sub("^--file=", "", script_arg))
source(file.path(here, "experiment-inputs.R"), local = TRUE)
audit <- experiment_verify_inputs(args[[1L]], normalizePath(args[[2L]], mustWork = TRUE))
write.csv(audit$summary, file.path(args[[2L]], "input-change-audit.csv"), row.names = FALSE, quote = TRUE)
if (!is.null(audit$fixed)) {
  write.csv(audit$fixed, file.path(args[[2L]], "fixed-rate-audit.csv"), row.names = FALSE, quote = TRUE)
}
cat("Verified exact experiment inputs for ", args[[1L]], ".\n", sep = "")
