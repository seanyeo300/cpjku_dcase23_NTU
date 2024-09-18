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

def run_multiple_scripts(scripts_with_args):
    try:
        for script_name, args in scripts_with_args:
            # Convert all arguments to strings
            args = [str(arg) for arg in args]
            subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names and arguments
    scripts_to_run = [
        ('run_passt_SKD_Cochl_TAU_h5.py', ['--lr', 1e-6, "--experiment_name", "NTU_SKD1_gen3_nmc8pby8_T_scdhurtv_S_1e-6_h5"]),
        ('run_passt_SKD_Cochl_TAU_h5.py', ['--lr', 1e-6, "--experiment_name", "NTU_SKD1_gen3_nmc8pby8_T_scdhurtv_S_1e-6_h5"]),
        ('run_passt_SKD_Cochl_TAU_h5.py', ['--lr', 1e-6, "--experiment_name", "NTU_SKD1_gen3_nmc8pby8_T_scdhurtv_S_1e-6_h5"])
    ]
    run_multiple_scripts(scripts_to_run)