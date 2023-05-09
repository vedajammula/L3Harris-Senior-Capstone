import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="RUSSELL 2000", page_icon="📈")

st.markdown("# RUSSELL 2000")
tab1, tab2, tab3 = st.tabs(["Real RUSSELL2000 Data", "Simulated Attack RUSSELL2000 Without Cleaning","Simulated Attack RUSSELL2000 With Cleaning"])

filename = 'russel2000_all.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'
data_flag = 0

pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab1:
    pipeline.run_pipeline()

filename = 'new_russel.csv'
data_flag = 2
pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab2:
    pipeline.run_pipeline()

data_flag = 1
pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab3:
    pipeline.run_pipeline()