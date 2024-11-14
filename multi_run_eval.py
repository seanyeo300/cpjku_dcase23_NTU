# multi_run.py
import subprocess

def run_multiple_scripts(scripts_with_args):
    try:
        for script_name, args in scripts_with_args:
            subprocess.run(['python', script_name] + args)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example: Replace these with your actual script names and arguments 50+100%
    
    scripts_to_run = [
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', '8gpctett']),
        ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', 'kgh56uev']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', 'no72vxvm']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', 'emvp0ahe']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', 'rndo9pmo']),
        # ('run_passt_training_subsets_DIR_FMS_h5.py', ['--project_name',"NTU24_ASC",'--evaluate', '--ckpt_id', 'vmpdce38']),
    ]
    run_multiple_scripts(scripts_to_run)
# 5% FMS only 
# ["dbl1yun4", "brcaxnko", "z0xsdw9o", "59z6lxjj", "6ehbfn9i", "vj86j26r"]
# 5% DSIT FMS,DIR
# ["yyki5y1f", "fskag87u", "a7ms5l1f", "5acz12c2", "bxgn5l84", "jktyxl3l"]
# tv1b
["jiw5bohu", "erxj7yo6", "q5ct8wik", "1l9r0xw7", "z3448sj6", "hoo6924h"] #running now on single
#tv2c
["jiw5bohu", "erxj7yo6", "q5ct8wik","fskag87u","bxgn5l84", "yyki5y1f"] # no need to run, just ensemble once the first three are done
# TA (passt)
['rsxdobd5', 'b229ynyb', 'zc2tl07k']

#10% tv1b
["0mek3m3i","17bgq5xu","uczgumpm","duuzsrgh","uj36xuac","ypw9kvtd"]
# tv2c
["0mek3m3i","17bgq5xu","uczgumpm","ltq74ut8","kwe5kw2y","b259dd8e"] # no need to run, just ensemble once the first three are done
# tv3
["ltq74ut8","kwe5kw2y","b259dd8e","yoknz8lp","t269df7z","ppsps225"]

#25%    
# tv1
# ["vbkz6eb4", "327kbswi", "u9kbvlz3", "m0792a81", "ssezo41p", "z3kv9ew0"] 
# # tv2
# ["vbkz6eb4","u9kbvlz3", "ssezo41p", "n5ntbunr", "5u2cths8","7xljnu1s"]
# tv1b
["s2rf9v2j", "55osep77", "bmfe7xg7", "zg9dcrr3", "w5muk32o","axgvrvk1"]
#tv2c
["s2rf9v2j", "55osep77", "bmfe7xg7", "me35cqrb", "gvdbvn5e","no72vxvm"]
# tv3
["me35cqrb", "gvdbvn5e","no72vxvm", "emvp0ahe" ,"rndo9pmo", "vmpdce38"]

#50%
# # tv1
# ["cw4t0bco", "jobstdfr", "a7k2mviq", "y5j0nvj2", "duqnsqa9", "j96nxtpb"]
# # tv2
# ["cw4t0bco", "a7k2mviq", "duqnsqa9", "kzi3j8i9", "bn9vp2m4", "jhjnxe4z"]
# tv1b
["fzcw72bp","hy0mzx6z","1nqhp83v","q8rxd8a3","hxpe9ycz","403jqv4g"]
# tv2c
["fzcw72bp","hy0mzx6z","1nqhp83v","jpuww899","bneij1tp","bu9ogmxo"]
# tv3
["jpuww899", "bneij1tp", "bu9ogmxo","15cmvyi2", "ejso4chn", "m70acb6q"]

#100%
# tv1b
["um60xoby","5glbzaab","r3etgxpp","gnwkrjpe","8jnf4ele","kcazd8oh"]
# tv2c
["um60xoby","5glbzaab","r3etgxpp","qupf3zrc","e45ivt1f","1sevgzul"]
# tv3
["qupf3zrc","e45ivt1f","1sevgzul","a0dnfaoh","pghlf5fv","rcze3lr8"]


#10%
["buzkwfs9","5xerh8xn","l9vftqos"]
#5%
["f5hhbj59","o661pbve","a27p3f3e"]