# N-Gram Language Models for Turkish



## Introduction

N-gram language models are statistical models used in natural language processing (NLP) to predict the likelihood of a sequence of words or characters. In these models, an "n-gram" is a contiguous sequence of 'n' items (such as words, syllables, or characters) in a given text. N-gram models analyze text sequences to predict the probability of each item, based on the frequency of previous items in the sequence. These models are valuable in various NLP tasks, including speech recognition, text generation, and language modeling.




### Syllable-Based vs. Character-Based Models

This study explores two distinct approaches to modeling Turkish text using N-gram models: syllable-based and character-based. Each method has unique strengths and limitations, particularly relevant to Turkish's linguistic structure:

- **Syllable-Based Model**: Turkish, being an agglutinative language, heavily relies on suffixation to form words. A syllable-based model aligns well with this structure by capturing meaningful sub-word components, making it likely to capture semantic and syntactic relationships in a more coherent way. For example, a syllable-based model could recognize recurring syllables that denote specific grammatical functions, improving its ability to model Turkish's morphologically rich structure. 
- **Character-Based Model**: Unlike the syllable-based approach, character-based models treat individual characters as the basic unit of analysis. While this might overlook the semantic context of syllables, it can capture unique morphological patterns across word boundaries, offering benefits in low-resource scenarios or for modeling highly inflected forms. This approach is particularly useful in languages with complex inflectional patterns or orthographic variations.




### Objectives of the Homework 

The main objectives of this homework assignment include: 

1. **Implementation of N-Gram Models**: Develop both syllable-based and character-based N-gram models for Turkish, with configurations for 1-gram, 2-gram, and 3-gram variations. 
2. **Smoothing Techniques**: Apply appropriate smoothing techniques to adjust probabilities for unseen n-grams in each model. 
3. **Evaluation Metrics**: Evaluate the models using perplexity—a common metric in NLP that measures how well a language model predicts a given sequence. 
4. **Sentence Generation**: Generate random sentences using both models to observe the output quality across different configurations and assess the models' fluency and coherence.



### Expected Outcomes

 Given Turkish's linguistic characteristics, it is anticipated that: 

- **Syllable-based models** will outperform character-based models in terms of perplexity, as syllables capture more meaningful linguistic units in Turkish. Higher-order syllable-based models (e.g., 2-gram or 3-gram) are expected to provide more coherent sentence structures due to their ability to capture longer sequences and contextual dependencies. 

  

- **Character-based models** may produce lower perplexity values at the 1-gram level due to the prevalence of high-frequency individual letters. However, for higher-order n-grams, this approach may suffer from increased perplexity due to the lack of semantic coherence across sequences of letters. 

  

  Through this approach, the assignment aims to determine which model and n-gram configuration is most suitable for accurately representing Turkish text data. The results will shed light on whether syllable-based modeling is superior for an agglutinative language like Turkish, or if character-based modeling provides comparable performance across certain tasks. By comparing expected and actual outcomes, the study seeks to evaluate the strengths and limitations of each model in the context of Turkish language modeling.





## Design and Implementation

In this section, we will detail the project structure, Makefile usage, and the main functionality of the `main.py` file. This will provide a foundational understanding before diving into the specifics of each module in the following steps.




### Project Structure and General Workflow

The project directory is organized to handle various stages, from data processing to model generation and evaluation:

- **data/**: This directory is split into two subdirectories:
  - **raw/**: Contains the unprocessed `wiki_00.txt` file, which is the primary data source.
  - **processed/**: Stores the cleaned and prepared data files, including syllable and character representations in both training and testing formats. Files such as `wiki_00_syllables_train.txt` and `wiki_00_characters_test.txt` are saved here for streamlined access during model training and evaluation.

- **modules/**: Contains modular Python scripts, each responsible for distinct processes:
  - `data_preparation.py`: Responsible for data cleaning and preparation.
  - `split_data.py`: Splits processed data into training and testing sets.
  - `ngram_calculation.py`: Builds and saves n-gram tables with frequency counts.
  - `perplexity.py`: Calculates perplexity values for model evaluation.
  - `text_gen.py`: Generates random sentences based on n-gram tables.
  - `turkish_syllable.py`: Handles Turkish-specific syllable processing tasks.
  - `tables.py`: Generate tables for sample sentences and perplexity data.
  
- **results/**: Stores outputs generated by the models, including n-gram frequency tables for syllable- and character-based models at different levels, such as `character_ngram_1-gram.txt` and `syllable_ngram_2-gram.txt`.




### Makefile

The `Makefile` is a key component for automating various steps in the workflow. Each target in the `Makefile` is designed to simplify complex or repetitive tasks, ensuring a consistent setup and execution:

- **install**: Installs required libraries listed in `requirements.txt`.
- **clean**: Initiates data cleaning and processing, creating syllable- and character-based files for training and testing.
- **ngram**: Generates n-gram models and saves them in the `results` directory. It leverages the `--ngram` argument with `main.py`.
- **perplexity**: Calculates perplexity for each model and each n-gram level, aiding in model evaluation.
- **textgen**: Produces random sentences using trained n-gram models, executed by calling `main.py` with the `--textgen` argument.
- **clear**: Deletes all files from `data/processed` and `results`, resetting the data for fresh processing.

The `Makefile` allows for efficient control over the entire workflow, providing users with quick access to various functions and enabling them to replicate results effortlessly.




### Explanation of `main.py`

The `main.py` file is the central script, coordinating data processing, model training, perplexity calculations, and text generation. Using the `argparse` module, `main.py` offers four main options:

1. **Data Cleaning (`--clean`)**:
   - When invoked with `--clean`, this command preprocesses the raw data file and produces syllable- and character-based outputs in the `data/processed` directory.
   - It calls the `process_text_file()` function for cleaning, then splits the cleaned data into training and test sets using `split_data()`.

2. **N-gram Model Generation (`--ngram`)**:
   - This command, triggered with `--ngram`, first checks for processed data files with `check_processed_data_exists()`.
   - It then generates n-gram tables (1-gram, 2-gram, and 3-gram) for both syllable and character data. The `calculate_and_save_ngrams()` function builds n-gram frequency tables and applies Good-Turing smoothing before saving the tables in the `results` directory.

3. **Perplexity Calculation (`--perplexity`)**:
   - When `--perplexity` is specified, this command calculates the perplexity of syllable and character models on the test data.
   - Using the `calculate_perplexity()` function, it computes the perplexity score for each n-gram level, helping to assess model effectiveness.

4. **Text Generation (`--textgen`)**:
   - This command generates random sentences based on syllable and character models. The `generate_random_sentence()` function uses the n-gram tables to create sentences of specified lengths, providing an insight into the model’s language generation capabilities.

The modular and well-defined structure of `main.py` orchestrates the entire project workflow, ensuring that each part functions cohesively to achieve the overall project goals.



**Main Function:**

![image-20241027135241530](/home/ahmet/.config/Typora/typora-user-images/image-20241027135241530.png)



#### Cleaning and Preprocessing (`--clean` Flag)

The `--clean` flag initiates the data cleaning and preprocessing pipeline, which is crucial for transforming raw text data into syllable-based and character-based formats suitable for n-gram model training. The operations are as follows:



1. **File Paths**:   
   - The `raw_file` variable points to the main data source (`./data/raw/wiki_00.txt`).   
   - The `syllable_output` and `character_output` variables define output files for syllable-based and character-based processed text (`wiki_00_syllables.txt` and `wiki_00_characters.txt`). 
2. **Data Processing**:   
   - The `process_text_file()` function is called twice, first with `model_type="syllable"` and then with `model_type="character"`. This function, found in `data_preparation.py`, handles all text cleaning, normalization, and format transformations needed for each model type.   
   - For syllable-based processing, it syllabifies words and preserves punctuation marks. For character-based processing, it splits text into individual characters, separated by spaces. 
3. **Data Splitting**:   
   - After processing, the `split_data()` function is called to partition the data into training (95%) and testing (5%) subsets. This helps ensure that n-gram models can be trained on one subset and evaluated on another.   
   - The function outputs `wiki_00_syllables_train.txt` and `wiki_00_syllables_test.txt` for syllable-based data, and `wiki_00_characters_train.txt` and `wiki_00_characters_test.txt` for character-based data.



##### Detailed Function Descriptions

###### `data_preparation.py`
The `data_preparation.py` script includes various text preprocessing steps designed to clean and normalize Turkish language data, particularly for Wikipedia text, which may contain HTML tags, abbreviations, and non-Turkish characters.


- **`process_text_file(file_path, output_file_path, model_type="syllable")`**:
  	- This main function orchestrates the entire cleaning and processing workflow, operating line by line on the input file.
  	- It uses the `tqdm` library to provide progress feedback, especially useful for large datasets.
  	- It invokes `process_text()` to handle specific cleaning steps, such as removing links, converting numbers to words, and syllabifying or character-segmenting based on `model_type`.
  	- **Output**: The processed content is saved to `output_file_path`.

![image-20241027135952265](/home/ahmet/.config/Typora/typora-user-images/image-20241027135952265.png)



- **`process_text(content, model_type="syllable")`**:  	

  - Executes multiple cleaning steps in sequence:    

    1. **Link Removal**: Uses a regex pattern to strip out URLs.    

    2. **Special Space Replacement**: Converts non-breaking spaces to regular spaces.    

    3. **HTML Tag Removal**: Filters out HTML and Wikipedia tags based on an extensive `html_tags` list and a regex pattern.   

    4. **Non-Turkish Character Removal**: Ensures only Turkish characters and punctuation are retained.    

    5. **Abbreviation Expansion**: Expands common abbreviations (e.g., `örn.` to `örneğin`).  

    6. **Number Conversion**: Transforms numeric values into their Turkish word equivalents (e.g., `3` to `üç`).    

    7. **Lowercasing**: Converts uppercase letters to lowercase, respecting Turkish character mappings.    

    8. **Segmentation**: Based on `model_type`, calls either `syllabify_text_with_punctuation()` or `char_based_text()` for syllable or character segmentation.  

       **Output**: Returns the cleaned and segmented text for line-by-line processing.

![image-20241027140253997](/home/ahmet/.config/Typora/typora-user-images/image-20241027140253997.png)



- **Helper Functions**:  
  - **`remove_non_turkish_characters(content)`**: Filters out characters that are not commonly used in Turkish text.  
  - **`replace_turkish_characters(content)`**: Normalizes Turkish characters by mapping them to basic Latin equivalents (optional).  
  - **`clean_html_tags(content)`**: Removes HTML tags while preserving text.  
  - **`expand_abbreviations(content)`**: Uses a dictionary to replace common abbreviations with their full forms.  
  - **`syllabify_text_with_punctuation(content)`**: Applies syllable segmentation while preserving punctuation.  
  - **`char_based_text(content)`**: Splits text into individual characters separated by spaces.





###### `split_data.py` 
The `split_data.py` script uses Scikit-Learn’s `train_test_split` to partition data files into training and testing subsets. 

- **`split_data(file_path, train_file_path, test_file_path, test_size=0.05)`**:  
  - **Input**: Reads a text file, then splits it based on the specified `test_size` (5% for testing by default).  
  - **Output**: Writes the resulting training and test data to `train_file_path` and `test_file_path`.  
- Helper functions `read_file()` and `write_file()` handle file I/O, including line reading and writing, to prevent any character encoding issues.

![image-20241027140635954](/home/ahmet/.config/Typora/typora-user-images/image-20241027140635954.png)





###### `turkish_syllable.py` 

The `turkish_syllable.py` script contains functions specific to Turkish syllabification, which is essential for segmenting Turkish words accurately. I couldn't find an adequate library for this part, so I wrote it all by myself.

- **`syllabify(word)`**:
  - **Description**: Segments Turkish words into syllables based on a set of phonological rules.
  - **Logic**: Implements rule-based syllabification by checking for vowels and consonants and handling syllable boundaries (e.g., `sessiz - sessiz - sesli` patterns).
  - **Output**: Returns a list of syllables for each word, maintaining Turkish phonological rules.

- **`syllabify_text_with_punctuation(content)`**:
  - **Description**: Uses `syllabify(word)` to process each word in a sentence while preserving punctuation as separate tokens.
  - **Output**: Returns a string where words are replaced by their syllable-separated forms and punctuation is retained.



![image-20241027141039422](/home/ahmet/.config/Typora/typora-user-images/image-20241027141039422.png)





#### N-gram Model Generation (`--ngram` Flag)

The `--ngram` flag is responsible for generating n-gram models for both syllable-based and character-based data using the processed training data. The n-grams produced help in calculating the probabilities required for the language model, and Good-Turing smoothing is applied to account for unseen n-grams, improving the model’s accuracy.




1. **Checking Processed Data**:
   - The function `check_processed_data_exists()` ensures that the necessary preprocessed data files exist in the `processed` directory (`wiki_00_syllables_train.txt` and `wiki_00_characters_train.txt`). If they are absent, a message prompts the user to run `--clean` first.

2. **Defining File Paths**:
   - Two main files serve as inputs for this step: 
     - `syllable_train_file` for syllable-based n-gram calculations.
     - `character_train_file` for character-based n-gram calculations.
   - Output file prefixes are set for each n-gram type, such as `syllable_ngram_` and `character_ngram_`, which allow the n-gram files to be stored in a structured way.

3. **N-gram Table Generation**:
   - The code iterates over values of `n` (from 1 to 3) for unigram, bigram, and trigram models:
     - **Syllable-based n-grams**: The `calculate_and_save_ngrams()` function is called with the syllable-based training data and output prefix for each `n`.
     - **Character-based n-grams**: The same function is called with character-based data and prefix.

4. **Completion Confirmation**:
   - Upon completion of the loop, a message is printed indicating successful generation of n-gram models.



![image-20241027141500399](/home/ahmet/.config/Typora/typora-user-images/image-20241027141500399.png)



##### Detailed Function Descriptions

****

**`calculate_and_save_ngrams(file_path, output_prefix, n)`**:

This function is central to the `--ngram` flag’s functionality and performs the following steps:

- **Loading Data**:
  - It first loads data from `file_path` using `load_data()` to retrieve lines of preprocessed text.

- **Building N-gram Table**:
  - It then invokes `build_ngram_table()` with the data and n-value to generate the n-gram frequency table. This table is stored in a dictionary, where keys are n-grams (tuples of `n` tokens), and values are their occurrence counts in the text.

- **Applying Smoothing**:
  - The Good-Turing smoothing technique is applied via `apply_good_turing_smoothing()` to adjust the frequency of observed n-grams. Smoothing handles cases of zero-frequency n-grams (unseen n-grams) and produces more reliable probability estimates by adjusting observed frequencies.

- **Saving N-gram Table**:
  - The `save_ngram_table()` function stores the resulting smoothed n-grams and their probabilities in a specified output file (`output_path`). 



###### `ngram_calculation.py` 

This module contains various functions for building, smoothing, and saving/loading n-gram models.



\- **`generate_ngrams(text, n)`**:  

- **Description**: Converts the input text into a list of n-grams by iterating over tokens in the text. 

- **Logic**: The function splits the text into tokens (words or characters) and forms tuples of `n` tokens, ensuring that all overlapping n-grams are captured.  

- **Output**: Returns a list of n-gram tuples.



\- **`build_ngram_table(data, n)`**:  

- **Description**: Builds an n-gram frequency table and a frequency count dictionary.  
- **Logic**: For each line in `data`, the function splits tokens and generates n-grams. Each n-gram’s frequency is recorded in `ngram_table`, while `freq_count` counts the number of n-grams at each frequency level.  
- **Output**: Returns `ngram_table` with n-gram frequencies and `freq_count` for frequency distribution, both essential for Good-Turing smoothing.



\- **`apply_good_turing_smoothing(ngram_table, freq_count)`**:  

- **Description**: Adjusts n-gram frequencies using Good-Turing smoothing to handle unseen n-grams.  
- **Logic**: For each n-gram frequency `f`:    - If a higher frequency (f+1) exists in `freq_count`, the smoothed frequency is calculated as `(f+1) * (count of f+1) / (count of f)`.    - Otherwise, the original frequency is used.  
- **Normalization**: The smoothed frequencies are divided by the total n-gram count to calculate relative probabilities.  
- **Output**: Returns a dictionary of n-grams with their smoothed probabilities.



\- **`save_ngram_table(ngram_table, file_path)`**:  

- **Description**: Saves the n-gram table to a specified file.  
- **Logic**: Iterates over `ngram_table` and writes each n-gram and its smoothed frequency to a file, with each n-gram token separated by spaces.  
- **Output**: Creates an output file that stores n-grams with their smoothed frequencies for later use.



\- **`load_data(file_path)`**:  

- **Description**: Reads lines from a specified file path.  
- **Logic**: Opens the file in read mode and returns a list of lines (processed text).  
- **Output**: Returns the list of processed data lines, used as input for building n-grams.



\- **`load_ngram_table(file_path)`**:  

- **Description**: Loads an existing n-gram table from a file.  
- **Logic**: Reads each line in the file, splits the n-gram tokens, and parses the smoothed frequency value.  
- **Output**: Returns a dictionary representing the n-gram table, useful for generating text or calculating perplexity.





#### Perplexity Calculation (`--perplexity` Flag)

The `--perplexity` flag is designed to measure the effectiveness of the generated n-gram models in predicting test data. By calculating perplexity, we assess the uncertainty of the language model regarding the test data. Lower perplexity scores indicate a better model fit, making this metric an important indicator of model performance.




1. **Loading N-gram Tables**:
   - For each n-gram level (unigram, bigram, trigram), the code loads precomputed n-gram probabilities from previously saved files (`syllable_ngram_{n}-gram.txt` and `character_ngram_{n}-gram.txt`). These files are loaded into dictionaries (`syllable_ngram_tables` and `character_ngram_tables`) for syllable-based and character-based models.
   - The `load_ngram_table()` function retrieves these tables, containing n-grams and their smoothed probabilities. These probabilities are necessary for calculating the likelihood of each n-gram in the test data.

2. **Loading Test Data**:
   - The test data for both syllable-based (`wiki_00_syllables_test.txt`) and character-based (`wiki_00_characters_test.txt`) models is loaded from the `processed` directory using `load_data()` to provide sentences for perplexity calculation.

3. **Calculating Perplexity**:
   - For each n-gram level (1 to 3), `calculate_perplexity()` computes perplexity values separately for syllable-based and character-based models, iterating over each sentence in the test data.
   - Perplexity is calculated by accumulating log-probabilities for all n-grams in the test set and using these probabilities to compute the model’s uncertainty regarding unseen data.

4. **Output of Perplexity Scores**:
   - Perplexity values are printed for each n-gram model, enabling an easy comparison of the effectiveness of each model in predicting new data.



![image-20241027142609803](/home/ahmet/.config/Typora/typora-user-images/image-20241027142609803.png)




##### Detailed Function Descriptions

**`calculate_perplexity(test_data, ngram_probs, n)`**
The primary function for calculating perplexity operates as follows:

- **Input Parameters**:
  - `test_data`: List of strings, where each line represents a sentence in the test data.
  - `ngram_probs`: Dictionary of n-grams with their smoothed probabilities.
  - `n`: The n-gram level for which perplexity is being calculated (1 for unigram, 2 for bigram, etc.).

- **Logic**:
  - **Initialize Counters**: `total_log_prob` accumulates the log-probabilities for all n-grams, while `total_ngrams` counts the number of n-grams in the test set.
  - **Iterate Through Sentences**:
    - Each sentence is tokenized, and the function iterates over each possible n-gram in the sentence.
    - For each n-gram, it checks if the n-gram exists in the `ngram_probs` dictionary:
      - **Found**: Adds the log of the n-gram’s probability to `total_log_prob`.
      - **Not Found**: Assigns a very small probability (`1e-10`) to avoid zero probability, adding its log to `total_log_prob`.
    - Updates `total_ngrams` by counting each processed n-gram.
  - **Calculate Average Log Probability**: `avg_log_prob` is computed as `total_log_prob / total_ngrams`.
  - **Perplexity Calculation**: Using the formula `perplexity = exp(-avg_log_prob)`, the perplexity score is derived.

- **Output**:
  - Returns the perplexity score, which indicates the model’s performance on the test data. Lower values signify a better model.



##### Supporting Functions in `perplexity.py`

- **`load_ngram_probabilities(file_path)`**:
  - **Description**: Loads n-gram probabilities from a file, structured as n-grams with their probabilities on each line.
  - **Logic**: Reads each line, splits the n-gram and its probability, and stores it in a dictionary with the n-gram tuple as the key.
  - **Output**: Returns a dictionary of n-grams and their probabilities, used in perplexity calculations.




#### Random Sentence Generation (`--textgen` Flag)

The `--textgen` flag enables the system to generate random sentences by leveraging the trained n-gram models. This functionality is used to assess how well the models capture the structure and flow of the Turkish language by observing the coherency of generated sentences. For each model (syllable-based and character-based), sentences are generated for 1-gram, 2-gram, and 3-gram models, and displayed as output.

##### Steps Involved in Sentence Generation

1. **Loading N-gram Models**:
   - The code begins by loading n-gram models for both syllable-based and character-based configurations. Each n-gram model (1, 2, and 3) is loaded from saved files using `load_ngram_table()` and stored in dictionaries (`syllable_ngram_table` and `character_ngram_table`).
   - This setup allows the function `generate_random_sentence()` to access the precomputed probabilities for each n-gram during sentence generation.

2. **Generating Sentences**:
   - For each n-gram model (1 to 3), `generate_random_sentence()` is called, generating sentences with a maximum length of 50 tokens. These sentences are printed for both syllable-based and character-based models.
   - Each sentence generation leverages the n-gram probabilities to produce a sequence of tokens (syllables or characters), aiming to form coherent phrases.
   
   


##### Detailed Function Descriptions

**`generate_random_sentence(ngram_table, start_context=tuple(), max_length=15)`**

The primary function for generating random sentences works as follows:

- **Input Parameters**:
  - `ngram_table`: The n-gram model dictionary, where each n-gram has a probability or frequency score.
  - `start_context`: The starting context, which is initially an empty tuple. If empty, the function selects a frequent starting n-gram from the model.
  - `max_length`: The maximum number of tokens in the generated sentence.

- **Logic**:
  - **Selecting Start Context**:
    - If `start_context` is empty, the function defaults to a frequent initial n-gram (often a unigram or capitalized token) to start the sentence.
  - **Sentence Generation Loop**:
    - Using `current_context`, the function iterates over `max_length - len(start_context)`, appending one token per iteration based on the following:
      - **Top N-grams Selection**: Retrieves top probable n-grams using `get_top_n_grams()` based on `current_context`.
      - **Next Token Selection**:
        - If top n-grams are available, the function randomly selects a next n-gram using their probabilities as weights.
        - The chosen n-gram’s last token is appended to `sentence`, updating `current_context` with the latest tokens.
      - **Fallback**: If no suitable n-grams are found, the function selects a random n-gram from the model to continue sentence generation, which prevents the sentence from stalling.
  - **Output**:
    - After reaching `max_length`, the function returns `sentence` as a single string, combining tokens with spaces.
    
    

##### Supporting Functions in `text_gen.py`

- **`get_top_n_grams(ngram_table, current_context)`**:
  - **Description**: Retrieves the most probable n-grams from the model given the current context. This function is crucial for generating realistic sentences, as it identifies likely continuations of the sentence.
  - **Output**: Returns a list of probable n-grams, allowing `generate_random_sentence()` to construct the next part of the sentence with contextual relevance.





### A sample console output for `make all` command:

```bash
ahmet@ahmet-Inspiron-14-5401:~/DERSLER/4_SINIF/fall/NLP/hw1$ make all
<o> Installing dependencies...
python3 -m pip install --upgrade pip
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pip in /home/ahmet/.local/lib/python3.10/site-packages (24.2)
Collecting pip
  Downloading pip-24.3-py3-none-any.whl.metadata (3.7 kB)
Downloading pip-24.3-py3-none-any.whl (1.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 199.7 kB/s eta 0:00:00
Installing collected packages: pip
  Attempting uninstall: pip
    Found existing installation: pip 24.2
    Uninstalling pip-24.2:
      Successfully uninstalled pip-24.2
Successfully installed pip-24.3
python3 -m pip install -r requirements.txt
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: inflect==7.4.0 in /home/ahmet/.local/lib/python3.10/site-packages (from -r requirements.txt (line 1)) (7.4.0)
Requirement already satisfied: scikit_learn==1.5.2 in /home/ahmet/.local/lib/python3.10/site-packages (from -r requirements.txt (line 2)) (1.5.2)
Requirement already satisfied: tqdm==4.66.5 in /home/ahmet/.local/lib/python3.10/site-packages (from -r requirements.txt (line 3)) (4.66.5)
Requirement already satisfied: more-itertools>=8.5.0 in /usr/lib/python3/dist-packages (from inflect==7.4.0->-r requirements.txt (line 1)) (8.10.0)
Requirement already satisfied: typeguard>=4.0.1 in /home/ahmet/.local/lib/python3.10/site-packages (from inflect==7.4.0->-r requirements.txt (line 1)) (4.3.0)
Requirement already satisfied: numpy>=1.19.5 in /home/ahmet/.local/lib/python3.10/site-packages (from scikit_learn==1.5.2->-r requirements.txt (line 2)) (1.22.4)
Requirement already satisfied: scipy>=1.6.0 in /usr/lib/python3/dist-packages (from scikit_learn==1.5.2->-r requirements.txt (line 2)) (1.8.0)
Requirement already satisfied: joblib>=1.2.0 in /home/ahmet/.local/lib/python3.10/site-packages (from scikit_learn==1.5.2->-r requirements.txt (line 2)) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in /home/ahmet/.local/lib/python3.10/site-packages (from scikit_learn==1.5.2->-r requirements.txt (line 2)) (3.5.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /home/ahmet/.local/lib/python3.10/site-packages (from typeguard>=4.0.1->inflect==7.4.0->-r requirements.txt (line 1)) (4.12.2)
<o> Cleaning and processing data...
python3 main.py --clean
<-> 'modules' package installed.
<-> Running data cleaning and preprocessing...
Processing syllable  : 100%|████████████████████████████████████████████████████████████| 4547965/4547965 [04:57<00:00, 15299.92line/s]
Processing character : 100%|████████████████████████████████████████████████████████████| 4547965/4547965 [02:30<00:00, 30206.49line/s]
<-> Data extraction process begins
<-> File saved at: ./data/processed/wiki_00_syllables_train.txt
<-> File saved at: ./data/processed/wiki_00_syllables_test.txt
<-> File saved at: ./data/processed/wiki_00_characters_train.txt
<-> File saved at: ./data/processed/wiki_00_characters_test.txt
<-> Data cleaning and preprocessing completed.
<o> Generating N-gram models...
python3 main.py --ngram
<-> 'modules' package installed.
<-> Running n-gram model generation...
Building 1-Gram: 100%|██████████████████████████████████████████████████████████████████| 4320566/4320566 [00:45<00:00, 94063.56line/s]
Building 2-Gram: 100%|██████████████████████████████████████████████████████████████████| 4320566/4320566 [01:05<00:00, 66201.25line/s]
Building 3-Gram: 100%|██████████████████████████████████████████████████████████████████| 4320566/4320566 [01:32<00:00, 46636.89line/s]
Building 1-Gram: 100%|█████████████████████████████████████████████████████████████████| 8641133/8641133 [01:14<00:00, 115347.31line/s]
Building 2-Gram: 100%|█████████████████████████████████████████████████████████████████| 8641133/8641133 [01:21<00:00, 105491.91line/s]
Building 3-Gram: 100%|██████████████████████████████████████████████████████████████████| 8641133/8641133 [01:30<00:00, 95661.55line/s]
<-> N-gram model generation completed.
<o> Calculating perplexity...
python3 main.py --perplexity
<-> 'modules' package installed.
<-> Calculating perplexity for syllable-based model...
Syllable-based 1-gram perplexity: 401.96632274279864
Syllable-based 2-gram perplexity: 23452.647431934874
Syllable-based 3-gram perplexity: 443225.6714804
<-> Calculating perplexity for character-based model...
Character-based 1-gram perplexity: 23.11906113223346
Character-based 2-gram perplexity: 342.1483708751984
Character-based 3-gram perplexity: 3591.1622287487403
<o> Generating random sentences...
python3 main.py --textgen
<-> 'modules' package installed.
<-> Generating random sentences for syllable-based model...
Syllable-based 1-gram sentence: . wank jüt ğizz pvp dgi vuz rcv bocy jokk kjk trabz yelt kös flug Xju narr mrut bcm fraw vcp Qcd dağs ybus çdh tzu krc sıW sting rkb hjın dçn jnin thesp şvi şer tlip vejl trips bçl söyl cj hpn tsz buQ lmem kröp drü prl hşimy
Syllable-based 2-gram sentence: Wil li kas t o luş tu ru lu nan , ka ra sı na da ha zi ne km . bu nun da ha re tim o la rı na li a dı . bu ra sın da , ka ra fın dan ya da ya ' nın e
Syllable-based 3-gram sentence: Win dow s dar ta şen to yat dı ler ' la ğın dir nik ri crip ler ken yiz nir sın hod re rı o cham nal ji de tan ğı pir ) be be f re ca ley and bu mis ul ilk , ret ri le ği
<-> Generating random sentences for character-based model...
Character-based 1-gram sentence: a z o c : d . p c W o ) n r n j ' w e a : d ç n ö ; W r t z g Q u : ş s o ; b h f u t b X a u ? w z
Character-based 2-gram sentence: W i ö r i s ı l a k a n a n d a r a k t i s e n d a k t e n d ı n d i r i n d a k a r e l a l i r i
Character-based 3-gram sentence: W i l m q j W k c ş e ü a r ı X ' ö i i d ! b v n g : W d g s / j n x p ü e ' ç d ö r b ı ) k p o Q
ahmet@ahmet-Inspiron-14-5401:~/DERSLER/4_SINIF/fall/NLP/hw1$ 
```





## Results and Tables

tabloları falan ekle csv dosyalarından 
