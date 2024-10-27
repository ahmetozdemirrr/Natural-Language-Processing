# Python interpreter
PYTHON = python3


# File paths
REQUIREMENTS = requirements.txt
RAW_FILE = ./data/raw/wiki_00.txt
SYLLABLE_OUTPUT = ./data/processed/wiki_00_syllables.txt
CHARACTER_OUTPUT = ./data/processed/wiki_00_characters.txt


# Default target: install and clean up dependencies
all: install clean ngram perplexity textgen


# Target to install dependencies
install:
	@echo "\033[31m<o> Installing dependencies...\033[0m"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)


# Target to clean and process data
clean:
	@echo "\033[31m<o> Cleaning and processing data...\033[0m"
	$(PYTHON) main.py --clean


# Target to generate N-gram models
ngram:
	@echo "\033[31m<o> Generating N-gram models...\033[0m"
	$(PYTHON) main.py --ngram


# Target to calculate perplexity
perplexity:
	@echo "\033[31m<o> Calculating perplexity...\033[0m"
	$(PYTHON) main.py --perplexity
	

# Target to generate random sentences
textgen:
	@echo "\033[31m<o> Generating random sentences...\033[0m"
	$(PYTHON) main.py --textgen


# Target to remove all processed data files
clear:
	@echo "\033[31m<o> Removing all processed data...\033[0m"
	rm -rf ./data/processed/*.txt
	rm -rf ./results/*.txt
	rm -rf perplexity_results.csv
	rm -rf sample_sentences.csv


# Help messages for available targets
help:
	@echo "\033[31m<o>  Available targets:\033[0m"
	@echo "\033[31m<o>  make all        - Install dependencies and clean data.\033[0m"
	@echo "\033[31m<o>  make install    - Install dependencies.\033[0m"
	@echo "\033[31m<o>  make clean      - Clean and process data.\033[0m"
	@echo "\033[31m<o>  make ngram      - Generate N-gram models.\033[0m"
	@echo "\033[31m<o>  make perplexity - Calculate and display perplexity.\033[0m"
	@echo "\033[31m<o>	make textgen	- Generate random sentences using n-gram models.\033[0m"
	@echo "\033[31m<o>  make clear      - Remove all processed data.\033[0m"
