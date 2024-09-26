# multi_run.py

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
    script_name = 'run_passt_cochl_FT_subsets_DIR_FMS_h5.py'
    # script_name = 'run_passt_cochl_tau_slowfast_subsets_DIR_FMS_h5.py'
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ["--subset", "10", "--dir_prob", "0.6", "--mixstyle_p", "0.4"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [

        ("utupypwc", "NTU_PaSST_FTcs_FTtau_wk50wxro_sub10_FMS_DIR_fixh5"),       #SeqFT 1e-4 
        ("0r39k52v", "NTU_PaSST_FTcs_FTtau_qrrag30b_sub10_FMS_DIR_fixh5"),       #SeqFT 1e-5 
        ("6ip7syrn", "NTU_PaSST_FTcs_FTtau_7qghtor2_sub10_FMS_DIR_fixh5"),       #SeqFT 1e-6
        ("wk50wxro", "NTU_PaSST_SLcs_FTtau_wk50wxro_sub10_FMS_DIR_fixh5"),       #SL 1e-4 SIT
        ("qrrag30b", "NTU_PaSST_SLcs_FTtau_qrrag30b_sub10_FMS_DIR_fixh5"),       #SL 1e-5 SIT
        ("7qghtor2", "NTU_PaSST_SLcs_FTtau_7qghtor2_sub10_FMS_DIR_fixh5")        #SL 1e-6 SIT 
        # ("wk50wxro", "NTU_PaSST_SLcs_SLtau_wk50wxro_sub10_FMS_DIR_fixh5"),       #SL 1e-4 DSIT (half complete, see
                                                                                   #NTU_passt_SLcs_SLtau_wk50wxro_sub10_FMS_DIR_PretrainedStudent_fixh5)
        # ("qrrag30b", "NTU_PaSST_SLcs_SLtau_qrrag30b_sub10_FMS_DIR_fixh5"),       #SL 1e-5 DSIT
        # ("7qghtor2", "NTU_PaSST_SLcs_SLtau_7qghtor2_sub10_FMS_DIR_fixh5")        #SL 1e-6 DSIT

    ] 
    
    # Number of times to repeat each experiment
    num_repeats = 6

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    
# import subprocess

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
#         # ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         # ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"]),
#         ('run_passt_KD_Cochl_TAU_FT_subsets_DIR_FMS_h5_multirun_copy.py', [ "--subset", "5", "--dir_prob", "0.6","--ckpt_id", "fskag87u", "--experiment_name", "NTU_KD_Var3b-T_DSIT-S_FMS_DIR_sub5_fixh5"])
#     ]
#     run_multiple_scripts(scripts_to_run)