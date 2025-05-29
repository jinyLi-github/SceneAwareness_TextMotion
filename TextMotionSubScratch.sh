# Run the command for Stage 1 
python tools/locate_target.py -c configs/locate/locate_chatgpt.yaml

# Run the command for Stage 2 
#python tools/generate_results.py -c configs/test/generate.yaml


# if you want to run the code and find if there are any bugs, run this:
# qlogin -pe smp 8 -l h_vmem=11G -l gpu=1 -l gpu_type=ampere -l h_rt=1:0:0 -l rocky

