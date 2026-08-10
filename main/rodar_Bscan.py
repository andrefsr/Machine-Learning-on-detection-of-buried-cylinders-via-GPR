import os

N = 5

for i in range(1,N+1):

    texto = f"""
#title: Bscan {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: z 0.100 0.170 0 my_ricker
#rx: 0.080 0.170 0
#src_steps: 0.002 0 0
#rx_steps: 0.002 0 0

#geometry_objects_read: 0 0 0 solo_geom{i}.h5 solo_geom{i}_materials.txt
"""
    with open(f"Bscan{i}_.in", "w") as f:
        f.write(texto)

    os.system(f"python -m gprMax Bscan{i}_.in -n 60")

for j in range(1,N+1):
    os.system(f"python -m tools.outputfiles_merge Bscan{j}_ --remove-files")
    os.remove(f"Bscan{j}_.in")

#os.system(f"python -m tools.plot_Bscan Bscan{1}__merged.out Ez") 

