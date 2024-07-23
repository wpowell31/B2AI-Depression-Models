# B2AI-Depression-Models

This repositiory contains code for a depression modeling using machine learn from audio and clinical data part of the Bridge2AI-Voice project.
 - https://www.b2ai-voice.org/


## Running code files

1.  Open Visual Studio Code in the B2AI remote desktop workspace and activate the virtual environment in the terminal.
```
conda activate bridge2ai
```

2. Install relevant packages
```
pip install xgboost
```
```
pip install catboost
```

3. Process the data
```
python processv02.py
```
4. Run depression modeling with machine learning
```
python xgb2.1.py
```
```
python cat2.py
```

Running thee scripts will produce the outputs that are included in this repository.

By Will Powell
