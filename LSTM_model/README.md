Project Structure: 
- root directory: stock_data, LSTM_model, readme, Pipfile, Pipfile.lock (both pipfiles are used to install the project dependencies)
  - stock_data: CSVs of data for NASDAQ, Russell 2000, S&P 500 AND the CSVs with "new" in front of the file represent the simulated adversarial attacks
  - LSTM_model: the root of this folder contains: Data_Manipulations module, LSTM_sim.py which is the LSTM stock close price predicition model, Pipeline.py   which is where all the code is ran
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

- First once you have git cloned the repo, cd into L3Harris-Senior-Capstone/
- This project is set to run in python 3.9.6 but pipenv should automatically detect this from the Pipfile
- First to set up pipenv type: "pipenv install"
- To activate the pip environment type: "pipenv shell"
- Now all dependecies should be installed and you can check this by running: "pipenv run pip freeze", make sure streamlit is listed
- Finally we can run the pipeline and see results in the UI dashboard: "streamlit run Pipeline.py"
- If you need to add dependencies to the project type: "pipenv install 'packagenanme'"
- Similarly if you need to uninstall dependencies in the project type: "pipenv uninstall 'packagenanme'"
  - If you add or delete a dependency make sure to commit and push the pipfile and pipfile lock
- IMPORTANT!!!: When you are done running pipenv make sure to close it with: "exit"



