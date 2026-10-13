import os

N = 4

for i in range(1,N+1):
    os.system(f"python -m tools.outputfiles_merge Bscan{i}_ --remove-files")
    os.remove(f"Bscan{i}_.in")
    os.remove(f"Bscan{i}_log.txt")