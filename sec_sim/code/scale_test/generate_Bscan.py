import os

Ni = 1
Nf = 10

for i in range(Ni,Nf+1):
    os.system(f"python -m tools.outputfiles_merge Bscan{i}_ --remove-files")
    os.remove(f"Bscan{i}_.in")
    os.remove(f"Bscan{i}_log.txt")