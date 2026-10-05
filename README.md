# AI Text Detector 🤖🔍

[![Streamlit App](https://ai-text-detector-9pzpqcja7bzzpjpkzglsqf.streamlit.app/)

An end-to-end Machine Learning pipeline and interactive web application built to classify text as either human-written or AI-generated. 

## Overview
This project demonstrates a complete classical NLP workflow. It processes raw text data, extracts features using TF-IDF vectorization, and classifies the text using a trained Logistic Regression model. The backend pipeline is wrapped in a lightweight Streamlit UI for real-time inference.

## Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **NLP Techniques:** TF-IDF Vectorization, Text Preprocessing
* **Frontend/UI:** Streamlit

## Methodology
1. **Data Preprocessing:** Cleaned and structured the dataset using Pandas.
2. **Feature Engineering:** Converted raw text into numerical features using Scikit-learn's `TfidfVectorizer` to capture stylistic and vocabulary frequencies.
3. **Model Training:** Trained a Logistic Regression classifier on the vectorized data.
4. **Evaluation:** Evaluated using standard classification metrics (Accuracy: [Insert Accuracy %]).

## Run it Locally
1. Clone the repository: `git clone https://github.com/NotWhiteTeethTeen/ai-text-detector.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

**Author:** Ali Alssanm
