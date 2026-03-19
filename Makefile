.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = GAMs
PYTHON_INTERPRETER = python3

DATA_DIR ?= ./data
LOGS_DIR ?= ./logs
RESULTS_MODEL_DIR ?= ./results_model
OOD_DATA_DIR ?= ./data_ood
MODEL_TYPE ?= ecmac
R2_QUARTILE ?= 3
DATASET ?= airfoil_self_noise

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files

run_ecmac_on_talent:
	cd ./talent_benchmark
	export PYTHONPATH=.:../src
	KERAS_BACKEND=jax ./train_model_classical_on_all_data_parallel.sh --model_type ecmac --dataset_path $(DATA_DIR) --logs_path $(LOGS_DIR) --results_model_path $(RESULTS_MODEL_DIR)

make_knnr_ood_datasets:
	export PYTHONPATH=./talent_benchmark:./src
	KERAS_BACKEND=jax ./create_ood_data_adversarial_based_parallel.sh --dataset_path $(DATA_DIR) --ood_dataset_path $(OOD_DATA_DIR)  --rect_search_iters 300 --k_ratio 0.9 --num_of_repetitions 5 --num_of_worsening_sets 20 --use_knr True

make_vsvr_ood_datasets:
	export PYTHONPATH=./talent_benchmark:./src
	KERAS_BACKEND=jax ./create_ood_data_adversarial_based_parallel.sh --dataset_path $(DATA_DIR) --ood_dataset_path $(OOD_DATA_DIR)  --rect_search_iters 300 --k_ratio 0.9 --num_of_repetitions 5 --num_of_worsening_sets 20 --use_knr False

run_model_on_ood:
	cd ./talent_benchmark
	export PYTHONPATH=.:../src
	KERAS_BACKEND=jax ./train_model_classical_on_all_ood_data_parallel_r2_based.sh --model_type $(MODEL_TYPE) --ood_dataset_path $(OOD_DATA_DIR) --logs_path $(LOGS_DIR) --results_model_path $(RESULTS_MODEL_DIR) --r2_quartile $(R2_QUARTILE)

create_rankings_vis:
	export PYTHONPATH=./talent_benchmark:./src
	KERAS_BACKEND=jax python ./src/analysis/rankings/visualise_new_models_rankings.py --data_dir $(DATA_DIR) --logs_dir $(LOGS_DIR)

create_spline_vis:
	export PYTHONPATH=./talent_benchmark:./src
	KERAS_BACKEND=jax python .src/analysis/spline_generation/extract_and_visualise_splines.py --data_dir $(DATA_DIR) --results_model_dir $(RESULTS_MODEL_DIR) --dataset $(DATASET)


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
