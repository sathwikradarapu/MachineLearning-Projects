# Machine Learning Projects

Welcome to the **Machine Learning Projects** repository! This repository contains two interesting ML applications:

## 🏏 IPL Victory Probability Estimator

This project predicts the probability of a team winning an IPL match based on live match inputs using machine learning techniques.

### Tech Stack:
- Python
- Scikit-learn
- Streamlit

### Features:
- Takes match situation inputs like current score, wickets left, overs remaining, etc.
- Estimates the probability of a team's victory in real-time.
- Interactive and responsive UI for easy use.

### Input Parameters:
- Batting team
- Bowling team
- Target score
- Current score
- Overs completed
- Wickets fallen
- Recent performance (last 5 overs)

## 🧀 ML Based Stroke Risk Prediction

This project predicts the risk of stroke in a patient based on health and lifestyle factors.

### Tech Stack:
- Python
- Scikit-learn
- Streamlit

### Features:
- User-friendly interface to input patient details.
- Predicts the probability of having a stroke.
- Provides immediate and interpretable feedback.

### Input Parameters:
- Age
- Gender
- Hypertension
- Heart Disease
- Marital Status
- Work Type
- Residence Type
- Average Glucose Level
- BMI
- Smoking Status

## 🚀 How to Run Locally

1. **Clone the repository**:

    ```bash
    git clone https://github.com/sathwikradarapu/MachineLearning-Projects.git
    ```

2. **Install the required libraries**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Navigate into the project directory** and run the desired application:

    ```bash
    streamlit run app.py
    ```

4. Ensure that the trained model files (like `model.pkl`) are present in the project folder.

Developed with ❤️ by **Sathwik Radarapu**
