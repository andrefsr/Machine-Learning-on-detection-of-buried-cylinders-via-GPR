'''
Parâmetro	                       Arenoso	Intermediário	Argiloso
Areia	                            0,80	    0,60	      0,35
Argila	                            0,05	    0,20	      0,40
Densidade aparente (g/cm³)	        1,55	    1,45	      1,30
Densidade das partículas (g/cm³)	2,66	    2,66	      2,66
Umidade mínima	                    0,02	    0,03	      0,08
Umidade máxima	                    0,15	    0,25	      0,30
'''

#soil_peplinski: 0.80 0.05 1.55 2.66 0.02 0.15 arenoso (1)
#soil_peplinski: 0.60 0.20 1.45 2.66 0.03 0.25 intermediario (2)
#soil_peplinski: 0.35 0.40 1.30 2.66 0.08 0.30 argiloso (3)

import os
import numpy as np
import pandas as pd

Ni = 1
Nf = 3 ### Número de solos

obj = []

for i in range(Ni,Nf+1):

    soil_type = np.random.choice(range(1,4))
    radius = np.random.choice(np.linspace(0.006,0.020,20))
    y = np.random.choice(np.linspace(0.080,0.130,20))
    x = np.random.choice(np.linspace(0.030,0.210,20))
    
    if soil_type == 1: ### SOLO ARENOSO
        sand = '0.80'
        clay = '0.05'
        dens = '1.55'
        u_min = '0.02'
        u_max = '0.15'

    elif soil_type == 2: ### SOLO INTERMEDIÁRIO
        sand = '0.60'
        clay = '0.20'
        dens = '1.45'
        u_min = '0.03'
        u_max = '0.25'

    elif soil_type == 3: ### SOLO ARGILOSO
        sand = '0.35'
        clay = '0.40'
        dens = '1.30'
        u_min = '0.08'
        u_max = '0.30'

    texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: {sand} {clay} {dens} 2.66 {u_min} {u_max} my_soil

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 my_soil my_fractal_box

#add_surface_roughness: 0 0.170 0 0.240 0.170 0.002 1.5 1 1 0.164 0.176 my_fractal_box

#cylinder: {x} {y} 0 {x} {y} 0.002 {radius} pec

#geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""

    with open(f"solo{i}.in", "w") as f:
        f.write(texto)

    os.system(f"python -m gprMax solo{i}.in")

    os.remove(f"solo{i}.in")
    os.remove(f"solo{i}.out")

    obj.append([soil_type,radius,y,x])

df_label = pd.DataFrame(obj)
df_label = df_label.rename(columns={0:"soil_type",1:'radius',2:'pos_y',3:'pos_x'}) 
df_label.to_csv(f'labels{Ni}-{Nf}.csv', index=False)