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
    # script_name = 'run_training_dev_h5.py'
    script_name = 'run_passt_cochl_tau_slowfast_subsets_DIR_FMS_h5.py'
    # script_name = 'run_passt_cochl_PT_FT_tau_subsets_h5.py'
    # script_name = 'run_training_DynMN_h5_PL.py'
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ['--gpu','[0]',"--subset", "50", "--dir_prob", "0.6", "--mixstyle_p", "0.4","--n_epochs", "25","--n_classes","13"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [

        # ("utupypwc", "NTU_PaSST_FTcs_FTtau_utupypwc_sub25_FMS_DIR_fixh5"),       #SeqFT 1e-4 
        # ("0r39k52v", "NTU_PaSST_FTcs_FTtau_0r39k52v_sub100_FMS_DIR_fixh5"),       #SeqFT 1e-5 
        # ("6ip7syrn", "NTU_PaSST_FTcs_FTtau_6ip7syrn_sub25_FMS_DIR_fixh5"),       #SeqFT 1e-6
        # ("wk50wxro", "NTU_PaSST_SLcs_FTtau_wk50wxro_sub25_FMS_DIR_fixh5"),       #SL 1e-4 SIT
        # ("qrrag30b", "NTU_PaSST_SLcs_FTtau_qrrag30b_sub25_FMS_DIR_fixh5"),       #SL 1e-5 SIT
        # ("7qghtor2", "NTU_PaSST_SLcs_FTtau_7qghtor2_sub25_FMS_DIR_fixh5")        #SL 1e-6 SIT 
        # ("wk50wxro", "NTU_PaSST_SLcs_SLtau_wk50wxro_sub5_DIR_fixh5"),             #SL 1e-4 DSIT 
        # ("qrrag30b", "NTU_PaSST_SLcs_SLtau_qrrag30b_sub25_FMS_DIR_fixh5"),       #SL 1e-5 DSIT
        # ("7qghtor2", "NTU_PaSST_SLcs_SLtau_7qghtor2_sub25_FMS_DIR_fixh5")        #SL 1e-6 DSIT
        # (None, "sBCBL_FTtau_sub10_FMS_DIR_fixh5")                              #PTau First model trained used mixup, have disabled default mixup for the next 5
        # ("g1gzf3te", "NTU_PaSST_SLcsOL_SLtau_g1gzf3te_sub5_FMS_DIR_50epoch_fixh5"),       #SL 1e-4 DSIT
        # ("v7q0fght", "NTU_PaSST_SLcsOL_SLtau_v7q0fght_sub5_FMS_DIR_fixh5"),
        # ("j98ea5ub", "NTU_PaSST_SLcsNoOL_SLtau_j98ea5ub_sub5_FMS_DIR_fixh5"),       #SL 1e-4 DSIT 
        ("nmioakwl", "NTU_PaSST_SLcs_sub10_SLtau_nmioakwl_sub50_FMS_DIR_fixh5"),       #SL 1e-4 DSIT 
        
        # (None, "tMN30_FTtau_32K_FMS_DIR_sub5_fixh5")                       #FTtau_noASpt 

    ] 
    
    # Number of times to repeat each experiment
    num_repeats = 1

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