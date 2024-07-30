# multi_run.py
import subprocess

def run_multiple_scripts(scripts_with_args):
    try:
        for script_name, args in scripts_with_args:
            subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names and arguments
    scripts_to_run = [
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--ckpt_id', 'f0oxywl3']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', [ '--ckpt_id', 'f0oxywl3']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', [ '--ckpt_id', 'f0oxywl3'])
    ]
    run_multiple_scripts(scripts_to_run)