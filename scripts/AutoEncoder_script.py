import subprocess
import sys
import os
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.models import load_model

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Set TensorFlow logging level to suppress INFO and WARNING messages
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress TensorFlow INFO and WARNING messages


# Load the model
selected_model = input("Choose AutoEncoder model (1 for model trained on data 'city_day.csv', 2 for model trained on data 'taiwan2015.csv'): ")
true_value_dict = {}

if selected_model == '1':
    model_path = 'SAVED_MODEL/autoencoder_model_city_day.h5'
    data_folder = 'input_files/city_day/MLP'
    true_value_dict = {0: 'Good',
                           1: 'Moderate',
                           2: 'Poor',
                           3: 'Satisfactory',
                           4: 'Severe',
                           5: 'Very Poor'}
    print("You choose the model trained on 'city_day'.")
elif selected_model == '2':
    model_path = 'SAVED_MODEL/autoencoder_model_taiwan.h5'
    data_folder = 'input_files/taiwan2015/MLP'
    true_value_dict = {0: 'Good', 
                           1: 'Moderate', 
                           2: 'Satisfactory'}
    print("You choose the model trained on 'taiwan'.")
else:
    print("Invalid selection. Please choose 1 or 2.")
    sys.exit(1)

MLP_model = load_model(model_path)

# Function to preprocess input data
def preprocess_input(text, selected):
    # Split the text into lines and then split each line into values
    lines = text.strip().split('\n')
    data = []
    for line in lines:
        values = line.split(',')
        # Convert values to float and append to data
        data.append([float(value) for value in values])
    # Convert data to numpy array
    data = np.array(data)
    # Reshape data based on the selected model
    
    data = data.reshape(1, -1)  # Reshape for model trained on data 1
      
    return data

# Function to make predictions
def predict(input_data):
    # Make predictions
    predictions = MLP_model.predict(input_data)
    return predictions

# Function to read input from text file
def read_input_from_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text

# Function to read input and true value from text file
def read_input_and_true_value_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        lines = content.strip().split('\n')  # Split content into lines
        
        # Assuming the true value is on the last line, extract it
        true_value = lines[-1].strip()
        
        # Assuming the input data is on the lines before the true value line, join them
        input_data = '\n'.join(lines[:-1]).strip()
        
    return input_data, true_value


# Main function
def main():
    # List available input files
    input_files = os.listdir(data_folder)
    print("Available input files:")
    for file in input_files:
        print(file)
    
    # Get user's choice of input file
    selected_file = input("Enter the name of the input file you want to use: ")
    
    # Validate user's choice
    if selected_file not in input_files:
        print("Invalid input file selection.")
        sys.exit(1)
    
    # Get the selected input file path
    file_path = os.path.join(data_folder, selected_file)
    print(f"You chose input file: {selected_file}")
    
    # # Read input from selected file
    # input_text = read_input_from_file(file_path)
    
    # Read input and true value from selected file
    input_text, true_value = read_input_and_true_value_from_file(file_path)
    
    # Preprocess input data
    input_data = preprocess_input(input_text, selected_model)
    
    # # Get the true value index
    # true_value_index = list(true_value_dict.values()).index(true_value)
    
    # Print  true values
    print("True class:", true_value)
    # Make predictions
    predictions = predict(input_data)
    predicted_class_index = np.argmax(predictions)
    print("Predicted class index:", true_value_dict[predicted_class_index])

    # Ask the user if they want to continue evaluating with the MLP script
    user_input = input("Would you like to continue evaluating with the current script - AutoEncoder ? (yes/no): ")

    # Check user's response
    if user_input.lower() == 'yes':
        # Execute the current script again
        subprocess.call(["python", __file__])
    elif user_input.lower() == 'no':
        # Execute the main evaluation script again
        subprocess.call(["python", "models_testing.py"])
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")
# Entry point of the script
if __name__ == "__main__":
    main()
