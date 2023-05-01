Project Structure: 
- root directory: stock_data, LSTM_model, readme
  - stock_data: CSVs of data for NASDAQ, Russell 2000, S&P 500 AND the CSVs with "new" in front of the file represent the simulated adversarial attacks
  - LSTM_model: the root of this folder contains: Data_Manipulations module, LSTM_sim.py which is the LSTM stock close price predicition model, Pipeline.py   which is where all the code is ran, PipFile for pipenv dependencies, Pipfile.lock for pipenv dependencies, requirements.txt which is used to install all   the dependencies in the project
    - Data Manipulations Module:
      1) Cross_Validate.py 
      2) Generate_Time_Intervals.py
      3) Hole_Detection.py
      4) Hurst.py
      5) KNN_unsupervised.py
      6) LOF.py
      7) Random_Data.py
      8) Time_Intervals.py
      9) Window.py
      10) Window_parse.py
      11) __init__.py
      12) wrong_data.py


Working with PipEnv:

1) If you have not ran the pipenv in this project yet and this is the first clone:
- First once you have git cloned the repo, cd into L3Harris-Senior-Capstone/LSTM_Model/
- Now this project is set to pipenv running python 3.9.6 so we have to use that version! Start the pip environment by running "pipenv shell --python 3.9.6"
- To install the dependencies from requirement file run: "pipenv install -r requirements.txt"
- Now all dependecies should be installed and you can check this by running: "pipenv run pip freeze", make sure streamlit is listed
- Now we can run the pipeline and see results in the UI dashboard: "streamlit run Pipeline.py"
- When you are done running pipenv make sure to close it with: "exit"


2) If you have already run pipenv in this project before:
- Everything is contained in a pipenv so we should not need to locally install dependencies
- If you need to install a dependency follow this format: pipenv install --dev "packagename"
  - Pip env needs to be running already before you install the dependency
  - this will automatically add the dependency to the pipfile and thus to the pipenv so everyone will have it (remember to push so the pipfile gets             updated)
  - please do not manually add a dependency to the pipfile because that might mess up the depdenencies for your own system and possibly others
- To actually run the code we can do this in two steps:
  - First when you are cd'd into the /LSTM_model folder, type: pipenv shell --python 3.9.6
  - Second, since you have already had dependencies installed in a pipenv in this folder, dependencies should already be installed and you can check this       by: "pipenv run pip freeze"
    - If packages are not installed you either need to reinstall or you have multiple environments colliding, so in the first case run: "pipenv install -r       requirements.txt" and if their are multiple environments try: "pipenv sync --dev" 
    - If neither of those work try: "pipenv install --dev"
  - Third, to see UI in localhost type: "streamlit run Pipeline.py"
  - Remember to always close your environment when you are done with development so type: "exit" 

