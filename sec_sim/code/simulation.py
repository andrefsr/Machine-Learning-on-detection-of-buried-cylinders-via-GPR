'''
python -m tools.plot_Bscan Bscan1__merged.out Ez
'''

import os
import sys
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# CONFIGURAÇÕES

Ni = 501
Nf = 1000
NUM_WORKERS = 4 # Ótimo
N_TRACES = 60

PASTA = Path.cwd()

# FUNÇÃO: GERA E EXECUTA UMA SIMULAÇÃO

def executar_simulacao(i):

    arquivo_h5 = PASTA / f"solo_geom{i}.h5"
    arquivo_mat = PASTA / f"solo_geom{i}_materials.txt"
    arquivo_in = PASTA / f"Bscan{i}_.in"
    arquivo_log = PASTA / f"Bscan{i}_log.txt"

    # Verifica se os arquivos do solo existem
    if not arquivo_h5.exists():
        return i, False, f"Arquivo ausente: {arquivo_h5.name}"

    if not arquivo_mat.exists():
        return i, False, f"Arquivo ausente: {arquivo_mat.name}"
    
                ### DESCENTRALIZAR A POSIÇÃO DA ANTENA!!!!
    texto = f"""
#title: Bscan {i}
#num_threads: 1
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#time_window: 3e-9

#waveform: ricker 1 1.5e9 my_ricker
#hertzian_dipole: z 0.100 0.170 0 my_ricker
#rx: 0.080 0.170 0
#src_steps: 0.002 0 0
#rx_steps: 0.002 0 0

#geometry_objects_read: 0 0 0 {arquivo_h5.name} {arquivo_mat.name}
"""

    try:
        arquivo_in.write_text(texto, encoding="utf-8")

        # Executa no mesmo ambiente Python do script
        with open(arquivo_log, "w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", "gprMax", str(arquivo_in), "-n", str(N_TRACES)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=str(PASTA)
            )

        return i, True, "Concluída"

    except Exception as erro:
        return i, False, str(erro)

# EXECUÇÃO PARALELA

if __name__ == "__main__":

    concluidas = 0
    falhas = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:

        tarefas = {executor.submit(executar_simulacao, i): i for i in range(Ni, Nf + 1)}

        for tarefa in as_completed(tarefas):
            i, sucesso, mensagem = tarefa.result()

            if sucesso: concluidas += 1
            else:
                falhas.append((i, mensagem))
                print(f"Falha no caso {i}: {mensagem}")

            total = concluidas + len(falhas)

            if total % 25 == 0 or total == Nf:
                print(f"Progresso: {total}/{Nf} | " f"Concluídas: {concluidas} | " f"Falhas: {len(falhas)}")

    # Salva o relatório para retomar os casos com falha
    with open("falhas_simulacoes.txt", "w", encoding="utf-8") as f:
        for i, mensagem in falhas:
            f.write(f"{i}: {mensagem}\n")

    print("\nExecução finalizada.")
    print(f"Concluídas: {concluidas}")
    print(f"Falhas: {len(falhas)}")
