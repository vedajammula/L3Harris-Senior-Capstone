Project Structure: 
- root directory: stock_data, LSTM_model, readme, Pipfile, Pipfile.lock (both pipfiles are used to install the project dependencies)
  - stock_data: CSVs of data for NASDAQ, Russell 2000, S&P 500 AND the CSVs with "new" in front of the file represent the simulated adversarial attacks
  - LSTM_model: the root of this folder contains: Data_Manipulations module, LSTM_sim.py which is the LSTM stock close price predicition model,                 Pipeline.py, pages folder, home.py
    - So basically home.py is run with streamlit and acts as the main page for the dashboard, and then the pages folder contains 3 files for NASDAQ, DJIA,       and RUSSELL. In each page file Pipeline.py is called passing in the stock index csv, start date, and end date. 
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

IMPORTANT!!!: 
- make sure pipenv is installed and up to date in your computer system, you can do this in one comamand which will either install/update pipenv or tell you   that you have the most up to date version. Type: "pip3 install pipenv" or "pip install pipenv"
- this project will not run if you your current python version is less than 3.5, however for the best results I suggest using python 3.6 or higher (we have tested with python 3.9 and 3.11 and it works)
  - this is because pipenv requires python 3.5 or later to function properly and if you have python 2 this will not be supported at all
- whatever python version 3.6+ you choose to use, make sure that it is actually installed and on your path OR you can use an interpreter in VScode that       3.6+
- Additionally, no other pipenv or virtualenvs can be running, this will lead to pipenv install errors to check this just type "pipenv --rm" which will       either remove any pip or virtual envs that are open OR it will tell you there is nothing open to remove
- Streamlit library causes issues with python versions less than 3.10, therefore an additional dependency of URLLib3 is added to the Pipfile with             certain version to support python versions below 3.10
- If you are running python 3.10 or higher the extra URLLib3 depdency will not install properly causing streamlit errors, therefore go into the Pipfile       and delete the line where URLLib3 is listed and re-run the pipenv install command and it should work
- So requirements to actually run this project: 
  1) have python 3.6 or later versions installed and in your path OR in an interpreter with VScode (check this with the command: "python --version")
  2) have pipenv installed and up to date which can be done with either "pip3 install pipenv" OR "pip install pipenv"
  3) make sure there are no running pip or virtual envs with the command "pipenv --rm"
  4) if all else fails and none of these steps work, you can manually "pip install {packagename}" for each package listed in Pipfile (there are not that        many)


TO RUN THE PROJECT WITH PIPENV: 

- First once you have git cloned the repo, cd into L3Harris-Senior-Capstone/
- To set up pipenv type: "pipenv install"
  - This will install all dependencies from the pipfile lock
- To activate the pip environment type: "pipenv shell"
  - This will open a pipenv bash in your current terminal
- Now all dependecies should be installed and you can check this by running: "pipenv run pip freeze", make sure streamlit is listed
- Finally we can run the pipeline on datasets and see results in the UI dashboard so first cd into LSTM_model and then type: "streamlit run Home.py"
  - If you get an AttributeError (i.e. AttributeError: module 'collections' has no attribute 'MutableMapping'), try deleting "urllib3 = "==1.2.2"" from the Pipfile and rerunning.
- If you need to add dependencies to the project type: "pipenv install 'packagenanme'"
- Similarly if you need to uninstall dependencies in the project type: "pipenv uninstall 'packagenanme'"
  - If you add or delete a dependency make sure to commit and push the pipfile and pipfile lock
- IMPORTANT!!!: When you are done running pipenv make sure to close it with: "exit"
  - If you do not close your pipenv then there will be environment issues when you try to re-run the pipenv install
- LASTLY: everytime you want to run dashboard you have to follow the steps above which are: 
  1) cd L3Harris-Senior-Capstone/ 
  2) pipenv install 
  3) pipenv shell
  4) cd LSTM_model
  5) streamlit run Home.py
  
  
- By Sarthak Shukla (let me know if you need any help running)



