import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="NASDAQ", page_icon="📈")

st.markdown("# NASDAQ")
tab1, tab2, tab3 = st.tabs(["Real NASAQ Data", "Simulated Attack NASAQ Without Cleaning","Simulated Attack NASAQ With Cleaning"])

filename = 'nasdaq_all.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'
data_flag = 0

pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab1:
    pipeline.run_pipeline()

filename = 'new_nasdaq.csv'
pipeline = Pipeline(filename, start_date, end_date, data_flag)

with tab2:
    pipeline.run_pipeline()

data_flag = 1
with tab3:
    pipeline.run_pipeline()
