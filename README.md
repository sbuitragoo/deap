# DEAP
Experiments under DEAP dataset for emotion recognition.


## Usage

### 1. Download the required data from DEAP dataset:

Got to the [official site](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and submit the request to obtain the full dataset.

### 2. Clone this repository:

    git clone https://github.com/sbuitragoo/deap.git

* Move to 'develop' branch (which is the most frequently updated):
    
    git checkout develop

### 3. Set Up your Work Space:

* Create a virtual environment ([Python 3.9](https://www.python.org/downloads/release/python-390/) recomended, should be installed on your sistem):

        python3.9 -m venv "virtual_environment_name"

* Install the required dependencies:

        pip install requirements.txt

### 4. Start Working:

* If you want to make feature extraction again by yourself:

        sudo rm -r eeg_features

    Then:

        python3 code/eeg_feature_extraction.py

* When features are ready, run:

        python3 code/feature_importance.py