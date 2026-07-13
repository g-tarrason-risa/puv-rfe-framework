# Makefile — regenerate the PUV analysis pipeline.
#
# DEVCONTAINER ONLY: the notebooks hardcode BASE = /workspaces/CODESPACE/,
# so this must be run inside the devcontainer, not on a host checkout.
#
# Your three choices map to targets (compose them freely, left to right):
#   1) remove all pickle files ....... make clean-pickles   (also clears .model_cache)
#   2) remove all results files ...... make clean-results
#   3) rerun all analyses ............ make rerun
#
# Common combinations:
#   make rerun                       # refresh everything from existing pickles (fast)
#   make clean-pickles rerun         # recompute models from scratch, then plots
#   make rebuild                     # all three: wipe pickles + results, full recompute
#
# Ordering is enforced: prep -> eda -> analyses -> supplementary. Any failing
# step halts the run (no half-regenerated, silently-inconsistent output).

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := help

PYTHON  ?= python
JUPYTER ?= jupyter

# Root is this Makefile's directory (works regardless of where make is invoked).
ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

INT_DIR     := $(ROOT)/data/int
RESULTS_DIR := $(ROOT)/results
CACHE_DIR   := $(ROOT)/.model_cache

PREP_NB := $(ROOT)/notebooks/prep/prep.ipynb
EDA_NBS      := $(sort $(shell find $(ROOT)/notebooks/eda -name '*.ipynb' 2>/dev/null))
ANALYSIS_NBS := $(sort $(shell find $(ROOT)/notebooks/analyses -name '*.ipynb' 2>/dev/null))
SUPP_SCRIPT  := $(ROOT)/notebooks/analyses/supplementary_outputs.py

# Execute a notebook in place, with no per-cell timeout (nested-CV cells are slow).
NBEXEC := $(JUPYTER) nbconvert --to notebook --execute --inplace \
          --ExecutePreprocessor.timeout=-1

.PHONY: help clean-pickles clean-results clean prep eda analyses supplementary rerun rebuild

help:
	@echo "PUV pipeline — targets (devcontainer only):"
	@echo "  clean-pickles   Remove data/int/*.pkl and .model_cache/"
	@echo "  clean-results   Remove all files under results/ (keeps the folder tree)"
	@echo "  clean           clean-pickles + clean-results"
	@echo "  prep            Execute the data-prep notebook"
	@echo "  eda             Execute the EDA notebooks"
	@echo "  analyses        Execute the 16 analysis notebooks"
	@echo "  supplementary   Regenerate the reviewer supplementary figures/tables"
	@echo "  rerun           Full chain: prep -> eda -> analyses -> supplementary"
	@echo "  rebuild         clean + rerun (wipe everything, recompute from scratch)"
	@echo ""
	@echo "Examples:"
	@echo "  make rerun                 # refresh from cached pickles"
	@echo "  make clean-pickles rerun   # recompute models, then plots"
	@echo "  make rebuild               # full clean rebuild"

# --- cleaning ---------------------------------------------------------------

clean-pickles:
	@echo ">> removing pickles ($(INT_DIR)/*.pkl) and cache ($(CACHE_DIR))"
	rm -f $(INT_DIR)/*.pkl
	rm -rf $(CACHE_DIR)

# Deletes every file under results/ but keeps the directory skeleton, so
# notebooks that savefig into results/<dataset>/ don't fail on a missing dir.
clean-results:
	@echo ">> removing all files under $(RESULTS_DIR)"
	@if [ -d "$(RESULTS_DIR)" ]; then find $(RESULTS_DIR) -type f -delete; fi

clean: clean-pickles clean-results

# --- pipeline stages (each halts on first error via .SHELLFLAGS -e) ----------

prep:
	@echo ">> [prep] $(PREP_NB)"
	$(NBEXEC) $(PREP_NB)

eda:
	@for nb in $(EDA_NBS); do echo ">> [eda] $$nb"; $(NBEXEC) "$$nb" || exit 1; done

analyses:
	@for nb in $(ANALYSIS_NBS); do echo ">> [analysis] $$nb"; $(NBEXEC) "$$nb" || exit 1; done

supplementary:
	@echo ">> [supplementary] $(SUPP_SCRIPT)"
	$(PYTHON) $(SUPP_SCRIPT)

# Recursive sub-makes run sequentially; a failing stage aborts the whole run.
rerun:
	$(MAKE) prep
	$(MAKE) eda
	$(MAKE) analyses
	$(MAKE) supplementary

rebuild:
	$(MAKE) clean
	$(MAKE) rerun
