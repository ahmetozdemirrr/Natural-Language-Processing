This project aims to model the Turkish language using n-gram language models, covering both syllable-based and character-based modeling approaches. The project includes data cleaning, n-gram model generation, perplexity calculation, and random sentence generation functionalities. This structure allows for evaluating which approach—syllable or character—is more effective for n-gram-based modeling of the Turkish language.

### Makefile Overview

The Makefile in this project automates various tasks and includes the following commands:

- **`make all`**: Installs dependencies, cleans and processes data, generates n-gram models, calculates perplexity, and generates random sentences.
- **`make install`**: Installs the required Python packages for the project.
- **`make clean`**: Cleans and processes raw data into formatted syllable and character files.
- **`make ngram`**: Generates syllable and character-based n-gram models and saves them to output files.
- **`make perplexity`**: Calculates perplexity values on test data to assess model performance.
- **`make textgen`**: Uses the n-gram models to generate random sentences for both syllable-based and character-based models.
- **`make clear`**: Deletes all processed data files, cleaning the project directory.
- **`make help`**: Lists available commands and their descriptions.

This Makefile enables efficient management of all project stages quickly and easily.
