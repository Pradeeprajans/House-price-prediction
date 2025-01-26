# House Price Prediction

This project is a web-based application built using Flask that predicts the price of a house based on various features such as the size of the house, the number of bedrooms, location, and other relevant factors. The application leverages machine learning models trained on historical real estate data to make accurate price predictions. Users can input their house details through a web form, and the app will use pre-trained models to predict the house price.

# Features

Price Prediction: Predicts the price of a house based on input features like area, number of rooms, location, and other factors influencing property prices.

Data Exploration & Preprocessing: Data is cleaned, missing values are handled, and relevant features are selected to improve model performance.

Interactive Web Interface: A clean, simple form where users input house details such as area, number of bedrooms, and location.

Real-Time Predictions: Once the form is submitted, the app processes the data through the machine learning model and returns the predicted price.

# Project Structure
app.py: The main Flask application file that runs the web server and handles routing. It includes endpoints for displaying the form, processing submissions, and serving price predictions.

model.py: Contains code for training the model, including preprocessing, feature selection, and model fitting.

data/: Contains the dataset(s) used for training and testing the model.

requirements.txt: Lists the dependencies required for the project.

notebooks/:

data_exploration.ipynb: Jupyter notebook for exploring and preprocessing the data.
model_training.ipynb: Jupyter notebook for training and evaluating the model.
templates/:

index.html: The HTML template for the main form where users input house details for prediction.
result.html: Displays the predicted house price based on the input data.
static/:

style.css: The CSS file for styling the web pages.

# How the Trained Model Files are Deployed
The machine learning model for house price prediction is saved as a Pickle (.pkl) file. This model is loaded into the Flask backend when the user submits the form.

## Model Deployment Workflow:
User Submission: The user enters house details through the form.
Data Preprocessing: The Flask application processes the input data to match the format expected by the model.
Model Prediction:
The preprocessed data is passed into the trained machine learning model.
The model makes a prediction based on the input features.
Result Display: The Flask app returns the predicted house price to the user via the result.html page.
The model is loaded at the start of the app and used for every prediction made by the user.

# How to Run the Web Application
Follow these steps to run the Flask application locally:

## 1. Clone the Repository
Clone this repository to your local machine:
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

## 2. Install Dependencies
Ensure you have Python installed. Then, install the necessary dependencies by running:
pip install -r requirements.txt

## 3. Run the Flask Application
To start the Flask development server, run the following command:
python app.py

## 4. Access the Application

After running the Flask application, open your browser and navigate to http://127.0.0.1:5000. You will see a form to input house details such as size, number of rooms, and location. Once you submit the form, the app will display the predicted house price.

Conclusion

This project offers a practical application for predicting house prices using machine learning techniques. By using Flask, the app enables users to interact with the model in real-time, providing predictions based on their inputs. The application can be further enhanced with more sophisticated models, additional features, or deployed on cloud platforms for broader access.
