import streamlit as st
from Pipeline import Pipeline

st.set_page_config(page_title="SimulatedAttackNASDAQ", page_icon="❗")

st.markdown("# SimulatedAttackNASDAQ")
st.sidebar.header("SimulatedAttackNASDAQ")

filename = 'new_nasdaq.csv'
start_date = '2010-01-04'
end_date = '2017-01-03'

pipeline = Pipeline(filename, start_date, end_date)
pipeline.run_pipeline()