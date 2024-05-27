# multi_run.py
import subprocess

def run_script_twice(script_name):
    try:
        # Run the script for the first time
        subprocess.run(['python', script_name])
        # Run the script for the second time
        subprocess.run(['python', script_name])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_script_twice('run_passt_training_subsets.py')
