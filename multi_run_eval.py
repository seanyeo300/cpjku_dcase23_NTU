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
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', '1iif8p36'])
        # ('run_passt_training_subsets_DIR_FMS.py', ['--evaluate', '--ckpt_id', 'rk677ntr'])
        # ('run_passt_training_subsets_DIR_FMS.py', ['--evaluate', '--ckpt_id', 'xa9sscwh'])
    ]
    run_multiple_scripts(scripts_to_run)
#25%
["vbkz6eb4","u9kbvlz3","ssezo41p"]
#10%
["buzkwfs9","5xerh8xn","l9vftqos"]
#5%
["f5hhbj59","o661pbve","a27p3f3e"]