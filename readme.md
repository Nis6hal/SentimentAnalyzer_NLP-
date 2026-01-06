# 📊 Sentiment Analyzer

A simple sentiment analysis web application built with Streamlit and scikit-learn. This app predicts the sentiment of text input using a trained machine learning model.

## Features

- Clean and intuitive web interface
- Real-time sentiment prediction
- Trained on custom dataset using Support Vector Machine (SVM)
- Text preprocessing with stopword removal

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── training.py               # Model training script
├── utils.py                  # Helper functions for preprocessing
├── requirements.txt          # Python dependencies
├── samples.csv               # Training dataset
├── sentiment_model.pkl       # Trained model (generated after training)
└── vectorizer.pkl           # Fitted vectorizer (generated after training)
```

## Installation

1. Clone this or download the files

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

First, train the model using your dataset:

```bash
python training.py
```

This will:
- Load and preprocess data from `samples.csv`
- Train an SVM classifier
- Save the trained model as `sentiment_model.pkl`
- Save the vectorizer as `vectorizer.pkl`

### Running the App

Launch the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser. Enter any text and click "Analyze" to get the predicted sentiment.






#
## P.s (In some cases, it might require you to use venv to run this.)
# what I did to run in my PC 
    python -m venv venv                                                         (setup virtual environment)

    venv\Scripts\activate.bat                                                   (activate virtual environment)

    pip install --upgrade pip setuptools wheel                                  (updated tools to access newest prebuilt wheels)

    pip install streamlit scikit-learn pandas matplotlib seaborn nltk           (installed requirements)

    python training.py                                                          (builds sentiment analysis model)

    streamlit run app.py                                                        (run web application)

#





## Dataset

The model is trained on `samples.csv`, which contain two columns:
- `Sentence`: The text data
- `Sentiment`: The sentiment label

## Requirements

- Python 3.13+
- streamlit==1.31.0
- scikit-learn==1.3.2
- pandas==2.1.4
- numpy==1.26.2

## Developer

**Developed By:** Nischal Bhandari  
School of Engineering (SOE), Pokhara University (PU)
