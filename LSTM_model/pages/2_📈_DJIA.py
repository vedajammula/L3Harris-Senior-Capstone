import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="DJIA", page_icon="📈")

st.markdown("# DJIA")
st.sidebar.header("DJIA")

filename = 'djia_2012.csv'
start_date = '2013-01-03'
end_date = '2019-01-02'

pipeline = Pipeline(filename, start_date, end_date)
pipeline.run_pipeline()

# st.write('IN PROGRESS')