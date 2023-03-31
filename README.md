# DEAP
Experiments under DEAP dataset for emotion recognition.


## Usage

### 1. Download the required data from DEAP dataset:

* Go to the [official site](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) and submit the request to obtain the full dataset.

* When you already have the data, create a folder called DEAP and add the information contained in data_preprocessed_python or just rename that folder.

### 2. Clone this repository:

* After having your data inside the deap folder, at the same location run:

    git clone https://github.com/sbuitragoo/deap.git

* Move to 'develop' branch (which is the most frequently updated):
    
    git checkout develop

At this point your working directory should look like this:
```
ANY_FOLDER_YOU_WANT
│    
│
└───DEAP
│   │   s01.dat
│   │   s02.dat
│   │   s03.dat
│   │   s04.dat
│   │   ...       
│   
└───deap
    │   README.md
    │   requirements.txt
    │   za_klasifikaciju.csv
    │   .gitignore
    └───code
    │   
    │   
    └───eeg_features
```

Where **DEAP** contains the data and **deap** represents this repository.

### 3. Set Up your Work Space:

* Create a virtual environment ([Python 3.9](https://www.python.org/downloads/release/python-390/) recomended, should be installed on your sistem):

        python3.9 -m venv "virtual_environment_name"

* Install the required dependencies:

        pip install requirements.txt

### 4. Start Working:

* To make data pre processing:

        python3 code/pre_process.py params --db PATH_TO_DB