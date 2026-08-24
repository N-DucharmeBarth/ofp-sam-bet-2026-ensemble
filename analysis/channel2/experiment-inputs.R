# Ad hoc full-path MFCL runs for Experiments A and B (paired-refit-decision.md)
# and the representative-member fixed-reporting-rate run.
#
# Reuses the low-level file-editing and audit helpers from
# scripts/ensemble-inputs.R unmodified. Adds two things the frozen ensemble
# pipeline has no need for:
#   1. Preparing a design row that is NOT one of the 100 drawn ensemble
#      members (arbitrary steepness/tau/K/M0/creep combinations, e.g. the
#      ensemble median).
#   2. Fixing specific tag-reporting-rate groups at a chosen value with their
#      active (estimated) flag turned off, instead of letting MFCL estimate
#      them -- the "fixed X-hat" manipulation Experiment B and the
#      representative-member run both need.
#
# The design table is analysis/channel2/experiments/experiment-design.csv,
# same column schema as design/model-draws.csv plus two: fixed_rep_groups
# (semicolon-separated group ids) and fixed_rep_values (semicolon-separated,
# same order). Both blank for a plain (nothing-fixed) run.

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
.experiment_here <- dirname(sub("^--file=", "", script_arg[[1L]]))
.experiment_repo <- normalizePath(file.path(.experiment_here, "..", ".."), mustWork = TRUE)
source(file.path(.experiment_repo, "scripts", "ensemble-inputs.R"), local = TRUE)

experiment_design_path <- file.path(.experiment_here, "experiments", "experiment-design.csv")

experiment_load_row <- function(model_id) {
  design <- read.csv(experiment_design_path, stringsAsFactors = FALSE, check.names = FALSE)
  row <- design[design$ensemble_id == model_id, , drop = FALSE]
  if (nrow(row) != 1L) stop("Unknown experiment run: ", model_id, call. = FALSE)
  row
}

experiment_fixed_spec <- function(row) {
  groups <- trimws(row$fixed_rep_groups)
  values <- trimws(row$fixed_rep_values)
  if (!nzchar(groups)) return(NULL)
  g <- as.integer(strsplit(groups, ";")[[1L]])
  v <- as.numeric(strsplit(values, ";")[[1L]])
  if (length(g) != length(v)) stop("fixed_rep_groups/fixed_rep_values length mismatch.", call. = FALSE)
  if (anyNA(g) || anyNA(v)) stop("fixed_rep_groups/fixed_rep_values contain a placeholder that was never filled in.", call. = FALSE)
  data.frame(group_id = g, value = v)
}

# Locate the "# tag fish rep group flags" block (99 rows x 33 cols of group
# ids), identical across every ensemble member because it is structural, not
# a design axis. Verified equal to the fitted PAR's rep_group matrix for two
# members (one per arm) before this was written.
experiment_group_matrix <- function(ini_path) {
  lines <- readLines(ini_path, warn = FALSE)
  rows <- ensemble_rows_after(lines, "^# tag fish rep group flags[[:space:]]*$", 99L)
  do.call(rbind, lapply(rows, function(i) as.integer(ensemble_fields(lines[[i]]))))
}

# Write `value` into every cell of the "# tag fish rep" block belonging to
# `group_id`, and 0 into the matching cells of "# tag_fish_rep active flags",
# so MFCL starts at that value and never estimates it away.
experiment_fix_group <- function(ini_path, group_matrix, group_id, value) {
  hit <- group_matrix == group_id
  if (!any(hit)) stop("Group id ", group_id, " has no cells in the group-flags block.", call. = FALSE)

  lines <- readLines(ini_path, warn = FALSE)
  rep_rows <- ensemble_rows_after(lines, "^# tag fish rep[[:space:]]*$", 99L)
  active_rows <- ensemble_rows_after(lines, "^# tag_fish_rep active flags[[:space:]]*$", 99L)
  val_str <- formatC(value, format = "f", digits = 4)

  for (r in seq_len(nrow(hit))) {
    cols <- which(hit[r, ])
    if (!length(cols)) next
    rep_fields <- ensemble_fields(lines[[rep_rows[[r]]]])
    active_fields <- ensemble_fields(lines[[active_rows[[r]]]])
    for (c in cols) {
      rep_fields[[c]] <- val_str
      active_fields[[c]] <- "0"
    }
    lines[[rep_rows[[r]]]] <- paste(rep_fields, collapse = " ")
    lines[[active_rows[[r]]]] <- paste(active_fields, collapse = " ")
  }
  writeLines(lines, ini_path, useBytes = TRUE)
  invisible(hit)
}

experiment_prepare_inputs <- function(model_id, output) {
  row <- experiment_load_row(model_id)
  repo <- .experiment_repo
  ensemble_verify_manifest(file.path(repo, "model"), file.path(repo, "model", "MANIFEST.sha256"))
  ensemble_source_hashes(repo)
  if (dir.exists(output) && length(list.files(output, all.files = TRUE, no.. = TRUE))) {
    stop("Output directory is not empty: ", output, call. = FALSE)
  }
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  status <- system2("cp", c("-a", paste0(file.path(repo, "model"), "/."), output))
  if (!identical(status, 0L)) stop("Failed to copy frozen Diagnostic inputs.", call. = FALSE)
  if (!file.copy(file.path(repo, "mfclo64"), file.path(output, "mfclo64"), overwrite = TRUE)) {
    stop("Failed to copy mfclo64.", call. = FALSE)
  }
  Sys.chmod(file.path(output, c("mfclo64", "doitall.sh")), mode = "0755")

  ini <- file.path(output, "bet.ini")
  doitall <- file.path(output, "doitall.sh")
  h_value <- format(as.numeric(row$steepness), digits = 17, scientific = TRUE)
  tau_value <- format(as.numeric(row$tag_tau), digits = 17, scientific = TRUE)
  tau_par_value <- format(as.numeric(row$tau_fish_pars4), digits = 17, scientific = TRUE)
  m_value <- format(as.numeric(row$lorenzen_log_intercept), digits = 17, scientific = TRUE)
  mixing_source <- file.path(repo, "sources", "mixing", row$tag_mixing_source_file)

  ensemble_replace_field(ini, "^# sv[(]29[)][[:space:]]*$", 1L, 1L, h_value)
  ensemble_replace_field(ini, "^# age_pars[[:space:]]*$", 5L, 1L, m_value)
  ensemble_replace_tag_column(ini, 1L, source_path = mixing_source)
  ensemble_replace_tag_reporting(ini, as.integer(row$tag_reporting_flag2))
  ensemble_replace_assignment(
    file.path(output, "model-inputs", "S0.90-F2.conf"), "STEEPNESS", h_value
  )
  ensemble_replace_assignment(
    file.path(output, "model-inputs", "S0.90-F2.conf"), "TAU", tau_value
  )
  ensemble_replace_assignment(
    file.path(output, "model-inputs", "S0.90-F2.conf"), "TAU_FISH_PARS4", tau_par_value
  )
  ensemble_replace_effort(
    file.path(output, "bet.frq"),
    file.path(repo, "sources", "effort-creep", row$effort_source_file)
  )

  fixed <- experiment_fixed_spec(row)
  if (!is.null(fixed)) {
    gm <- experiment_group_matrix(ini)
    for (i in seq_len(nrow(fixed))) {
      experiment_fix_group(ini, gm, fixed$group_id[[i]], fixed$value[[i]])
    }
  }

  lines <- readLines(doitall, warn = FALSE)
  hit <- grep("^[[:space:]]*expected = -2[.]54930339768360[[:space:]]*$", lines)
  if (length(hit) != 1L) stop("Could not locate the final M audit in doitall.sh.", call. = FALSE)
  lines[[hit]] <- paste0("  expected = ", m_value)
  writeLines(lines, doitall, useBytes = TRUE)
  ensemble_refresh_manifest(output)

  metadata <- row
  metadata$diagnostic_source_job <- 21641L
  metadata$diagnostic_source_commit <- "3abf0c64fb9b0c2d70b9c672dc7d9a655d3060d6"
  metadata$diagnostic_model <- "Diagnostic"
  metadata$input_status <- "prepared-and-verified"
  write.csv(metadata, file.path(output, "ensemble-metadata.csv"), row.names = FALSE, quote = TRUE)

  files <- list.files(output, recursive = TRUE, full.names = TRUE)
  files <- files[!file.info(files)$isdir]
  relative <- substring(files, nchar(output) + 2L)
  keep <- relative != "INPUTS.sha256"
  input_manifest <- data.frame(
    sha256 = vapply(files[keep], ensemble_sha256, character(1)),
    file = relative[keep], stringsAsFactors = FALSE
  )
  input_manifest <- input_manifest[order(input_manifest$file), ]
  write.table(input_manifest, file.path(output, "INPUTS.sha256"), row.names = FALSE, col.names = FALSE, quote = FALSE)

  audit <- experiment_verify_inputs(model_id, output)
  write.csv(audit$summary, file.path(output, "input-change-audit.csv"), row.names = FALSE, quote = TRUE)
  if (!is.null(audit$fixed)) {
    write.csv(audit$fixed, file.path(output, "fixed-rate-audit.csv"), row.names = FALSE, quote = TRUE)
  }
  cat("Prepared and verified ", model_id, ": ", row$model_label, "\n", sep = "")
  invisible(audit)
}

experiment_verify_inputs <- function(model_id, run_dir) {
  row <- experiment_load_row(model_id)
  repo <- .experiment_repo
  ensemble_verify_manifest(file.path(repo, "model"), file.path(repo, "model", "MANIFEST.sha256"))
  ensemble_source_hashes(repo)
  ensemble_verify_manifest(run_dir, file.path(run_dir, "INPUTS.sha256"))

  base <- file.path(repo, "model")
  path <- file.path(run_dir, "bet.ini")
  base_path <- file.path(base, "bet.ini")
  observed_h <- as.numeric(ensemble_value_after(path, "^# sv[(]29[)][[:space:]]*$", 1L, 1L))
  observed_m <- as.numeric(ensemble_value_after(path, "^# age_pars[[:space:]]*$", 5L, 1L))
  observed_slope <- as.numeric(ensemble_value_after(path, "^# age_pars[[:space:]]*$", 5L, 2L))
  if (abs(observed_h - as.numeric(row$steepness)) > 1e-12) stop("Steepness mismatch in bet.ini", call. = FALSE)
  if (abs(observed_m - as.numeric(row$lorenzen_log_intercept)) > 1e-12 || observed_slope != -1) {
    stop("Lorenzen M0 mismatch in bet.ini", call. = FALSE)
  }
  source_tags <- ensemble_tag_matrix(file.path(repo, "sources", "mixing", row$tag_mixing_source_file))
  actual_tags <- ensemble_tag_matrix(path)
  base_tags <- ensemble_tag_matrix(base_path)
  expected_flag2 <- ensemble_expected_tag_reporting(as.integer(source_tags[, 1L]), as.integer(row$tag_reporting_flag2))
  if (!identical(actual_tags[, 1L], source_tags[, 1L]) ||
      any(as.integer(actual_tags[, 2L]) != expected_flag2) ||
      !identical(actual_tags[, -(1:2), drop = FALSE], base_tags[, -(1:2), drop = FALSE])) {
    stop("Tag flags do not match the selected mixing/reporting inputs in bet.ini", call. = FALSE)
  }

  base_lines <- readLines(base_path, warn = FALSE)
  actual_lines <- readLines(path, warn = FALSE)
  h_row <- ensemble_rows_after(base_lines, "^# sv[(]29[)][[:space:]]*$", 1L)[[1L]]
  m_row <- ensemble_rows_after(base_lines, "^# age_pars[[:space:]]*$", 5L)[[5L]]
  tag_rows <- ensemble_rows_after(base_lines, "^# tag flags[[:space:]]*$", 98L)
  allowed <- setNames(lapply(tag_rows, function(x) c(1L, 2L)), as.character(tag_rows))
  allowed[[as.character(h_row)]] <- 1L
  allowed[[as.character(m_row)]] <- 1L

  fixed <- experiment_fixed_spec(row)
  fixed_report <- NULL
  if (!is.null(fixed)) {
    gm <- experiment_group_matrix(base_path)
    rep_rows <- ensemble_rows_after(base_lines, "^# tag fish rep[[:space:]]*$", 99L)
    active_rows <- ensemble_rows_after(base_lines, "^# tag_fish_rep active flags[[:space:]]*$", 99L)
    fixed_report <- vector("list", nrow(fixed))
    for (i in seq_len(nrow(fixed))) {
      g <- fixed$group_id[[i]]; v <- fixed$value[[i]]
      hit <- gm == g
      touched_rep <- 0L; touched_active <- 0L
      wrong_value <- character(0)
      for (r in seq_len(nrow(hit))) {
        cols <- which(hit[r, ])
        if (!length(cols)) next
        rr <- rep_rows[[r]]; ar <- active_rows[[r]]
        allowed[[as.character(rr)]] <- unique(c(allowed[[as.character(rr)]], cols))
        allowed[[as.character(ar)]] <- unique(c(allowed[[as.character(ar)]], cols))
        rep_fields <- ensemble_fields(actual_lines[[rr]])
        active_fields <- ensemble_fields(actual_lines[[ar]])
        for (c in cols) {
          touched_rep <- touched_rep + 1L
          if (abs(as.numeric(rep_fields[[c]]) - v) > 5e-5) wrong_value <- c(wrong_value, rep_fields[[c]])
          touched_active <- touched_active + 1L
          if (active_fields[[c]] != "0") stop("Active flag not zeroed for group ", g, " at row ", r, " col ", c, call. = FALSE)
        }
      }
      if (length(wrong_value)) stop("Fixed rate not written correctly for group ", g, ": found ", paste(wrong_value, collapse = ","), call. = FALSE)
      if (touched_rep == 0L) stop("Group ", g, " matched no cells -- wrong group id?", call. = FALSE)
      fixed_report[[i]] <- data.frame(group_id = g, value = v, n_cells = touched_rep)
    }
    fixed_report <- do.call(rbind, fixed_report)
  }
  ensemble_assert_lines(base_path, path, allowed)

  model_input_rel <- file.path("model-inputs", "S0.90-F2.conf")
  model_input <- file.path(run_dir, model_input_rel)
  base_model_input <- file.path(base, model_input_rel)
  observed_model_h <- as.numeric(ensemble_assignment_value(model_input, "STEEPNESS"))
  observed_tau <- as.numeric(ensemble_assignment_value(model_input, "TAU"))
  observed_tau_par <- as.numeric(ensemble_assignment_value(model_input, "TAU_FISH_PARS4"))
  if (abs(observed_model_h - as.numeric(row$steepness)) > 1e-12 ||
      abs(observed_tau - as.numeric(row$tag_tau)) > 1e-12 ||
      abs(observed_tau_par - as.numeric(row$tau_fish_pars4)) > 1e-12 ||
      abs(observed_tau - (1 + exp(observed_tau_par))) > 1e-12 ||
      ensemble_assignment_value(model_input, "MODEL_ID") != "S0.90-F2" ||
      ensemble_assignment_value(model_input, "SELECTIVITY_MODEL") != "F2" ||
      ensemble_assignment_value(model_input, "SELECTIVITY_INPUT") != "selectivity-models/F2.csv") {
    stop("Job 21641 model-input identity or ensemble steepness mismatch.", call. = FALSE)
  }
  model_input_lines <- readLines(base_model_input, warn = FALSE)
  model_change_rows <- c(
    grep("^STEEPNESS=", model_input_lines), grep("^TAU=", model_input_lines),
    grep("^TAU_FISH_PARS4=", model_input_lines)
  )
  ensemble_assert_lines(
    base_model_input, model_input,
    setNames(rep(list(1L), length(model_change_rows)), as.character(model_change_rows))
  )

  base_frq <- readLines(file.path(base, "bet.frq"), warn = FALSE)
  actual_frq <- readLines(file.path(run_dir, "bet.frq"), warn = FALSE)
  source_frq <- readLines(file.path(repo, "sources", "effort-creep", row$effort_source_file), warn = FALSE)
  base_records <- ensemble_effort_records(base_frq)
  actual_records <- ensemble_effort_records(actual_frq)
  source_records <- ensemble_effort_records(source_frq)
  if (length(base_records) != 1458L || !setequal(names(base_records), names(actual_records)) ||
      !setequal(names(base_records), names(source_records))) stop("Incomplete F29-F33 effort records.", call. = FALSE)
  allowed_frq <- list()
  for (key in names(base_records)) {
    before <- base_records[[key]]$fields
    actual <- actual_records[[key]]$fields
    source <- source_records[[key]]$fields
    if (!identical(actual[[6L]], source[[6L]]) || !identical(before[-6L], actual[-6L])) {
      stop("Effort-creep transfer mismatch at ", key, call. = FALSE)
    }
    allowed_frq[[as.character(base_records[[key]]$line)]] <- 6L
  }
  ensemble_assert_lines(file.path(base, "bet.frq"), file.path(run_dir, "bet.frq"), allowed_frq)

  base_doitall <- readLines(file.path(base, "doitall.sh"), warn = FALSE)
  actual_doitall <- readLines(file.path(run_dir, "doitall.sh"), warn = FALSE)
  allowed_doitall_rows <- grep("^[[:space:]]*expected = -2[.]54930339768360[[:space:]]*$", base_doitall)
  allowed_doitall <- setNames(lapply(allowed_doitall_rows, function(x) seq_along(ensemble_fields(base_doitall[[x]]))),
                                as.character(allowed_doitall_rows))
  ensemble_assert_lines(file.path(base, "doitall.sh"), file.path(run_dir, "doitall.sh"), allowed_doitall)
  if (!any(grepl(paste0("expected = ", format(as.numeric(row$lorenzen_log_intercept), digits = 17, scientific = TRUE)),
                 actual_doitall, fixed = TRUE))) {
    stop("Final M audit was not updated in doitall.sh.", call. = FALSE)
  }

  base_manifest <- read.table(file.path(base, "MANIFEST.sha256"), col.names = c("sha256", "file"), stringsAsFactors = FALSE)
  actual_manifest <- read.table(file.path(run_dir, "MANIFEST.sha256"), col.names = c("sha256", "file"), stringsAsFactors = FALSE)
  if (!identical(base_manifest$file, actual_manifest$file)) stop("Model manifest file list changed.", call. = FALSE)
  permitted <- c("bet.ini", "bet.frq", "doitall.sh", model_input_rel)
  changed_manifest <- base_manifest$file[base_manifest$sha256 != actual_manifest$sha256]
  if (length(setdiff(changed_manifest, permitted))) stop("Unexpected model file changed: ", paste(setdiff(changed_manifest, permitted), collapse = ", "), call. = FALSE)
  for (relative in setdiff(base_manifest$file, permitted)) {
    if (ensemble_sha256(file.path(base, relative)) != ensemble_sha256(file.path(run_dir, relative))) {
      stop("Unexpected non-target input change: ", relative, call. = FALSE)
    }
  }

  metadata <- read.csv(file.path(run_dir, "ensemble-metadata.csv"), stringsAsFactors = FALSE, check.names = FALSE)
  if (nrow(metadata) != 1L || metadata$ensemble_id != model_id) stop("Ensemble metadata mismatch.", call. = FALSE)

  invisible(list(
    summary = data.frame(
      ensemble_id = model_id, model_label = row$model_label,
      steepness = row$steepness, tag_mixing_k_cutoff = row$tag_mixing_k_cutoff,
      tag_reporting_flag2 = row$tag_reporting_flag2, tag_tau = row$tag_tau,
      tau_fish_pars4 = row$tau_fish_pars4, m0_quarterly = row$m_age40_quarterly,
      effort_creep_primary = row$effort_creep_primary, effort_creep_secondary = row$effort_creep_secondary,
      status = "passed", stringsAsFactors = FALSE
    ),
    fixed = fixed_report
  ))
}
