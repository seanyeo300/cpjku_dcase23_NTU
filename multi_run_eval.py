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
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', 'yyki5y1f']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', 'fskag87u']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', 'a7ms5l1f']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', '5acz12c2']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', 'bxgn5l84']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--evaluate', '--ckpt_id', "jktyxl3l"])
    ]
    run_multiple_scripts(scripts_to_run)
    
# 5% FMS only 
["dbl1yun4", "brcaxnko", "z0xsdw9o", "59z6lxjj", "6ehbfn9i", "vj86j26r"]
# 5% DSIT FMS,DIR
["yyki5y1f", "fskag87u", "a7ms5l1f", "5acz12c2", "bxgn5l84", "jktyxl3l"]


#25%    
["vbkz6eb4","u9kbvlz3","ssezo41p"]
#10%
["buzkwfs9","5xerh8xn","l9vftqos"]
#5%
["f5hhbj59","o661pbve","a27p3f3e"]