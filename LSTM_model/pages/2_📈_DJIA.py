import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="DJIA", page_icon="📈")

st.markdown("# DJIA")
tab1, tab2, tab3 = st.tabs(["Real DJIA Data", "Simulated Attack DJIA Without Cleaning","Simulated Attack DJIA With Cleaning"])

filename = 'djia_2012.csv'
start_date = '2013-01-03'
end_date = '2020-01-02'
data_flag = 0

pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab1:
    pipeline.run_pipeline()

filename = 'new_djdatafinal.csv'
data_flag = 2
pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab2:
    pipeline.run_pipeline()

data_flag = 1
pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab3:
    pipeline.run_pipeline()

# st.write('IN PROGRESS')