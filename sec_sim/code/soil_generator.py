import os
import numpy as np
import pandas as pd

N = 5 ### Número de solos

label_list = []

for i in range(1,N+1):

    tem_cilindro = np.random.rand() > 0.5

    if tem_cilindro:
        label = 1
    else:
        label = 0
    if label == 1:

        texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 my_soil my_fractal_box
#cylinder: 0.120 0.080 0 0.120 0.080 0.002 0.01 pec

    #geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""
    if label == 0:

        texto = f"""
#title: Solo {i}
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil

#fractal_box: 0 0 0 0.240 0.170 0.002 1.5 1 1 1 50 my_soil my_fractal_box

    #geometry_view: 0 0 0 0.240 0.210 0.002 0.002 0.002 0.002 solo{i} n
#geometry_objects_write: 0 0 0 0.240 0.210 0.002 solo_geom{i}
"""
    label_list.append(label)

    with open(f"solo{i}.in", "w") as f:
        f.write(texto)

    os.system(f"python -m gprMax solo{i}.in")

for j in range(1,N+1):
    os.remove(f"solo{j}.in")
    os.remove(f"solo{j}.out")

df_label = pd.DataFrame(label_list)
df_label = df_label.rename(columns={0:"Label"}) 
df_label.to_csv('labels.csv', index=False)