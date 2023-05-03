import streamlit as st

st.set_page_config(page_title="Home", page_icon="👋",)

st.sidebar.success("Select a dataset for analysis above.")

st.title("L3Harris Adversarial Data Attack Detection Machine Learning Model")

st.header("Welcome to the L3Harris Senior Capstone Project!")
st.subheader("To run our model and analyses please choose a dataset from the sidebar")
st.subheader("Features of our data manipulations include: simulating adversarial attacks, simulating continous retraining with window parsing, hole detection, local outlier factor, unsupervised learning KNN, and Hurst exponent.")

st.caption("Project Name: Colorado University at Boulder 2022-2023 Computer Science Capstone Project for L3Harris Technologies (LSTM Knowledge Transfer from Stock Markets to Orbital Data)")
st.caption("High-level Objective Statement: Our neural network model will predict historical stock market trends for DJIA, S&P 500, and NASDAQ. Using this existing model we will adapt it to fit the needs of continuous training, anomaly detection, protection against adversarial attacks, as well as explainability of model predictions in the use case of orbital data.")
st.caption("Background Information: L3Harris Technologies works alongside the US Space Force to track information about the orbits of satellites, rockets, and other space debris. In order to reduce the work orbital analysts have to do, L3Harris Technologies wants to develop a robust ML model that is resistant to different kinds of attacks by potential adversaries, that is tolerant to mistakes made when classifying incoming data, and that can explain decisions to orbital analysts that haven’t worked with the model.")
st.caption("Successful Outcome Statement: The final revised LSTM model for the stock market predictions will be explainable, robust, and can operate in adversarial environments. The model, as well as a testing injection framework can be transferred and applied to frameworks and conditions in which L3Harris customer’s ML models are developed and deployed within their own production environments.")
st.caption("Strategic Alignment: This project incorporates Machine Learning and Artificial Intelligence which is a growing field within the industry. Using these techniques, we are able to reduce outliers and create more accurate models. These models will be beneficial for the company given they give the outputs needed for success. Corporate strategies highlight using the most optimal solutions. Optimal solutions come from using models in Machine Learning such as NN to produce respectable outcomes. Throughout this project, we will demonstrate team work, organizational skills, and thus our knowledge in the field. By working in an agile development environment using Trello, we are able to stay aligned with one another and keep track of updates in the project. With weekly meetings with sponsors and TA’s we will be able to keep everyone in the loop and make sure we are on the right path. Along with this, this project will allow us to ask questions and understand the company’s strategies more.")
st.caption("Key Initiative Alignment: Key initiatives include creating a stand-in domain for the company’s own model for space orbit detection using stock market data. Data will be tracked and analyzed in a model similarly to provide applicable analysis of a model without access to the company’s sensitive data.")
