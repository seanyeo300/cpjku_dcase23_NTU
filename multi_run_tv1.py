# multi_run.py
import subprocess

# def run_multiple_scripts(scripts_with_args):
#     try:
#         for script_name, args in scripts_with_args:
#             subprocess.run(['python', script_name] + args)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Example: Replace these with your actual script names and arguments
#     scripts_to_run = [
#         ('run_passt_cochl_PT_mel_h5.py', ['--lr', 1e-4]),
#         ('run_passt_cochl_PT_mel_h5.py', [ '--lr', 1e-5]),
#         ('run_passt_cochl_PT_mel_h5.py', [ '--lr', 1e-6])
#     ]
#     run_multiple_scripts(scripts_to_run)
    
#     import subprocess

# def run_multiple_scripts(scripts_with_args):
#     try:
#         for script_name, args in scripts_with_args:
#             # Convert all arguments to strings
#             args = [str(arg) for arg in args]
#             subprocess.run(['python', script_name] + args)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Example: Replace these with your actual script names and arguments
#     scripts_to_run = [
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "zs2yso3b", "--experiment_name", "NTU_KD_Var2b-T_SIT-S_FMS_DIR_sub5_fixh5"])
#     ]
#     run_multiple_scripts(scripts_to_run)
    
import subprocess

def run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats):
    try:
        for ckpt_id, experiment_name in ckpt_experiment_pairs:
            # Update arguments with the current ckpt_id and experiment_name
            ckpt_id_arg = "None" if ckpt_id is None else ckpt_id
            args = base_args + ["--ckpt_id", ckpt_id_arg, "--experiment_name", experiment_name]
            
            # Run the script multiple times with the same arguments
            for _ in range(num_repeats):
                subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Define the script to run
    script_name = 'run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_tv1.py'
    
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ["--subset", "5", "--dir_prob", "0", "--mixstyle_p", "0.4"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [
        # ("fskag87u", "NTU_KD_Var1-T_DSIT-S_FMS_DIR_sub5_fixh5"), #DSIT
        # ("leguwmeg", "NTU_KD_Var1-T_SIT-S_FMS_DIR_sub5_fixh5"),  #SIT FMS DIR
        ("dbl1yun4", "NTU_KD_Var1-T_SIT-S_FMS_sub5_fixh5"),      #SIT FMS
        # ("lm7o54or", "NTU_KD_Var1-T_SeqFT-S_FMS_DIR_sub5_fixh5"),#SeqFT
        # ("f5hhbj59", "NTU_KD_Var1-T_FTtau-S_FMS_DIR_sub5_fixh5"),#FTtau
        # (None, "NTU_KD_Var1-T_PTas-S_FMS_DIR_sub5_fixh5")          #Ptau
    ]
    
    # Number of times to repeat each experiment
    num_repeats = 6

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    