So this branch of the project is where I have implemented a pipeline structure and converted the LSTM Model simulation into a class. 
The flow structure in terms of development should be as follows:
1) no more jupyter notebooks! when we are coding all files should be python class files 
2) files where we are working on the data manipulation should go into the folder called Data manipulations
3) We will then import the manipulation files into the Pipeline.py file which is located in the /LSTM_model folder
4) Now, we in the run_pipeline() function in Pipeline.py, you can instantiate your manipulation class and then call the manipulation function
  - The manipulation functions should in most cases take a pandas dataframe in, and return a pandas data frame which we can then pass into the LSTM_model simulation
  - We will need to edit the LSTM_model_simulation to take a data frame in, but that is easy and can be done once all manipulations are complete
5) If we follow this structure we should be all good


How to run the UI model:
  - Everything is contained in a pipenv so we should not need to locally install depencies
  - If you need to install a dependency follow this format: pipenv install "packagename"
    - this will automatically add the dependency to the pipfile and thus to the pipenv so everyone will have it
    - please do not manually add a dependency to the pipfile because that might mess up the depdenencies for your own system and possibly others
  - To actually run the code we can do this in two steps:
    - First when you are cd'd into the /LSTM_model folder, type: pipenv shell
    - Second, type: streamlit run Pipeline.py


Streamlit docs: https://docs.streamlit.io/library/api-reference
  - how to do different types of texts in streamlit: https://docs.streamlit.io/library/api-reference/text
  - how to display data (df's): https://docs.streamlit.io/library/api-reference/data
  - how to display graphs (streamlit graphs, matplotlib graphs): https://docs.streamlit.io/library/api-reference/charts
 
