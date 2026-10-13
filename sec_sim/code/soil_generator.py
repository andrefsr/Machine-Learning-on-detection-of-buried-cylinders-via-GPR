'''
Parâmetro	                       Arenoso	Intermediário	Argiloso
Areia	                            0,80	    0,60	      0,35
Argila	                            0,05	    0,20	      0,40
Densidade aparente (g/cm³)	        1,55	    1,45	      1,30
Densidade das partículas (g/cm³)	2,66	    2,66	      2,66
Umidade mínima	                    0,02	    0,03	      0,08
Umidade máxima	                    0,15	    0,25	      0,30
'''

#soil_peplinski: 0.80 0.05 1.55 2.66 0.02 0.15 arenoso
#soil_peplinski: 0.60 0.20 1.45 2.66 0.03 0.25 intermediario
#soil_peplinski: 0.35 0.40 1.30 2.66 0.08 0.30 argiloso

import os
import numpy as np
import pandas as pd

N = 5 ### Número de solos

n = np.random.choice(range(1,4),N)

for i in range(1,N+1):

    if n[i-1] == 1: ### SOLO ARENOSO

        texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: 0.80 0.05 1.55 2.66 0.02 0.15 arenoso

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 arenoso my_fractal_box

    #add_surface_roughness: 0 0 0.002 0.240 0.170 0.002 1.5 1 1 0.001 0.003 my_fractal_box

#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.01 pec

#geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""

    if n[i-1] == 2: ### SOLO INTERMEDIÁRIO

        texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: 0.60 0.20 1.45 2.66 0.03 0.25 intermediario

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 intermediario my_fractal_box

    #add_surface_roughness: 0 0 0.002 0.240 0.170 0.002 1.5 1 1 0.001 0.003 my_fractal_box

#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.01 pec

#geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""

    if n[i-1] == 3: ### SOLO ARGILOSO

        texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: 0.35 0.40 1.30 2.66 0.08 0.30 argiloso

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 argiloso my_fractal_box

    #add_surface_roughness: 0 0 0.002 0.240 0.170 0.002 1.5 1 1 0.001 0.003 my_fractal_box

#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.01 pec

#geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""

    with open(f"solo{i}.in", "w") as f:
        f.write(texto)

    os.system(f"python -m gprMax solo{i}.in")

    os.remove(f"solo{i}.in")
    os.remove(f"solo{i}.out")

df_label = pd.DataFrame(n)
df_label = df_label.rename(columns={0:"Label"}) 
df_label.to_csv('labels.csv', index=False)