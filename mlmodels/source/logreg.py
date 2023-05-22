import os

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

dir_path = os.path.dirname(os.path.realpath(__file__))


class MlModel:
    def __init__(self) -> None:
        self.df = pd.read_csv(dir_path + "/bank.csv", encoding="ISO-8859-1")
        # Define the label mapping

        self.df_num = self.to_df_num(self.df)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(self.df_num["category"])

        # Define the predictors (independent variables)
        X = self.df_num[["pos", "mod_p", "mod_n", "neg"]]

        # Normalize the input features using the same scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Add a constant term to the predictors
        X_scaled = np.column_stack((np.ones(len(X_scaled)), X_scaled))

        # Train a logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X_scaled, y_encoded)

    def to_df_num(self, df):
        category_mapping = {
            "Increase": 3,
            "Moderately Increase": 2,
            "Moderately Decrease": 1,
            "Decrease": 0,
        }
        df["category"] = self.df["Gain Category"].map(category_mapping)

        # Convert the target variable to numeric labels
        df["category"] = pd.Categorical(df["category"])
        def_dummy = pd.get_dummies(df["sentiment"])
        df["pos"] = def_dummy["Positive"]
        df["mod_p"] = def_dummy["Slightly Positive"]
        df["mod_n"] = def_dummy["Slightly Negative"]
        df["neg"] = def_dummy["Negative"]
        return df

    def predict(self, news):
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(news)
        compound_score = sentiment["compound"]

        if compound_score >= 0.2:
            sentiment_label = "Positive"
        elif compound_score < 0.2 and compound_score > -0.2:
            sentiment_label = "Slightly Positive"
        elif compound_score <= -0.2 and compound_score > -0.5:
            sentiment_label = "Slightly Negative"
        else:
            sentiment_label = "Negative"

        print(f"Sentiment: {sentiment_label}")

        # Prepare the input data for prediction
        new_data = {
            "pos": [1 if sentiment_label == "Positive" else 0],
            "mod_p": [1 if sentiment_label == "Slightly Positive" else 0],
            "mod_n": [1 if sentiment_label == "Slightly Negative" else 0],
            "neg": [1 if sentiment_label == "Negative" else 0],
        }
        df_new = pd.DataFrame(new_data)

        # Normalize the input features of the new dataframe using the same scaler
        X_new = df_new[["pos", "mod_p", "mod_n", "neg"]]
        X_new_scaled = self.scaler.transform(X_new)

        # Add a constant term to the predictors of the new dataframe
        X_new_scaled = np.column_stack((np.ones(len(X_new_scaled)), X_new_scaled))

        # Make predictions using the trained model for the new data
        y_pred_prob_new = self.model.predict_proba(X_new_scaled)

        # Convert the predicted probabilities to predicted classes
        y_pred_class_new = np.argmax(y_pred_prob_new, axis=1)

        # Decode the predicted classes using the label encoder
        predicted_labels_new = self.label_encoder.inverse_transform(y_pred_class_new)

        # Map label values to human-readable format
        label_mapping = {
            3: "Increase",
            2: "Moderately Increase",
            1: "Moderately Decrease",
            0: "Decrease",
        }
        predicted_labels_new = [label_mapping[label] for label in predicted_labels_new]

        # Create a dataframe with the predicted classes and probabilities for the new data
        df_pred_new = pd.DataFrame({"Predicted": predicted_labels_new})
        df_pred_new["Probability"] = y_pred_prob_new.max(axis=1) * 100

        # Print the dataframe with predicted classes and probabilities for the new data
        return f"The stock price will be {df_pred_new['Predicted'].iloc[0]} with a probability of {df_pred_new['Probability'].iloc[0]:.2f}%"
