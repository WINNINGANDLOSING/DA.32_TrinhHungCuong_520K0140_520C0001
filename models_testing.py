import os

def execute_script(model_number):
    scripts_folder = "scripts"
    script_files = {
        1: "MLP_script.py",
        2: "LSTM_script.py",
        3: "BLSTM_script.py",
        4: "GRU_script.py",
        5: "AutoEncoder_script.py"
    }

    script_name = script_files.get(model_number)
    if script_name:
        script_path = os.path.join(scripts_folder, script_name)
        if os.path.exists(script_path):
            os.system(f"python {script_path}")
        else:
            print("Error: Script not found.")
    else:
        print("Invalid model number.")

def main():
    print("Please select the model you wish to evaluate: ")
    print("Type 1 for MLP")
    print("Type 2 for LSTM")
    print("Type 3 for BiLSTM")
    print("Type 4 for GRU")
    print("Type 5 for MLP AutoEncoder")

    model_number = input("Enter the number corresponding to the model: ")
    try:
        model_number = int(model_number)
        execute_script(model_number)
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
