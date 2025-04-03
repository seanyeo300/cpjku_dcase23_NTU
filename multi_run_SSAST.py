# multi_run.py
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
    script_name = 'run_PANN_training_subsets_DIR_FMS_h5.py'
    
    # Base arguments (common to all runs, except experiment name and ckpt_id)
    base_args = ["--subset", "5", "--gpu", "[1]" ,"--n_epochs", "50" ,"--dir_prob", "0.6", "--mixstyle_p", "0.4", "--weight_decay", "0.001", "--lr", "2e-5", "--resample_rate", "44100"]
    
    # List of tuples containing checkpoint IDs and their corresponding experiment names
    ckpt_experiment_pairs = [
        # ("fskag87u", "NTU_KD_Var1-T_DSIT-S_FMS_DIR_sub5_fixh5"), #DSIT
        # ("leguwmeg", "NTU_KD_Var1-T_SIT-S_FMS_DIR_sub5_fixh5"),  #SIT FMS DIR
        # ("dbl1yun4", "NTU_KD_Var1-T_SIT-S_FMS_sub5_fixh5"),      #SIT FMS
        # ("lm7o54or", "NTU_KD_Var1-T_SeqFT-S_FMS_DIR_sub5_fixh5"),#SeqFT
        # ("f5hhbj59", "NTU_KD_Var1-T_FTtau-S_FMS_DIR_sub5_fixh5"),#FTtau
        (None, "NTU_PANNs_FTtau_441K_nofmin_FMS_DIR_1e-4_adamw_WD0.001_h5")          #Ptau
    ]
    
    # Number of times to repeat each experiment
    num_repeats = 6

    # Run the script with different checkpoint IDs and experiment names
    run_multiple_scripts(script_name, base_args, ckpt_experiment_pairs, num_repeats)
    