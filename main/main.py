from scipy.io import loadmat
import numpy as np
import os
import matplotlib.pyplot as plt
import control as ctrl

save_dir = '../graphics/'

# Carregar dados do arquivo .mat
mat = loadmat('../datasets/Dataset_Grupo1.mat')
struct_degrau = mat.get('TARGET_DATA____ProjetoC213_Degrau')
degrau = struct_degrau[:, 1]  # vetor coluna
tempo = struct_degrau[:, 0]  # vetor coluna
struct_saida = mat.get('TARGET_DATA____ProjetoC213_Saida')
saida = struct_saida[:, 1]  # vetor coluna

# Usando o teorema do valor final, calculamos o valor de K
ValorFinal = saida[-1]
AmplitudeDegrau = degrau[-1]
k = ValorFinal / AmplitudeDegrau

print("\nValor de k:", k)

# Calculando o atraso
theta = 0
tau = 0
for i in range(len(saida)):
    if saida[i] != 0 and theta == 0:
        theta = tempo[i - 1]

    if saida[i] >= (0.9821 * ValorFinal):
        tau = (tempo[i] - theta) / 4
        break

# Verificando se tau é negativo e corrigindo se necessário
if tau < 0:
    tau = abs(tau)
    print("Corrigindo valor negativo de tau:", tau)

print("Valor do atraso de transporte:", theta)
print("Valor da constante de tempo:", tau)

# Ziegler-Nichols em malha aberta
Kp_zn = 0.6 / k
Ti_zn = theta / 2
Td_zn = 0.5 * tau

# Criando o controlador PID com os parâmetros de Ziegler-Nichols
num_pid_zn = [Kp_zn * Td_zn, Kp_zn, Kp_zn / Ti_zn]
den_pid_zn = [1, 0]
PID_zn = ctrl.TransferFunction(num_pid_zn, den_pid_zn)

# Criando o sistema em série com os parâmetros de Ziegler-Nichols
sys_atraso = ctrl.tf([1], [tau, 1])
Cs_zn = ctrl.series(PID_zn, sys_atraso)

# Gerando a resposta ao degrau do sistema em malha fechada com Ziegler-Nichols
tempo_resposta_zn, resposta_zn = ctrl.step_response(ctrl.feedback(Cs_zn, 1))

# Calculando informações adicionais usando a função step_info
info_zn = ctrl.step_info(ctrl.feedback(Cs_zn, 1))
tempo_subida_zn = info_zn['RiseTime']
tempo_acomodacao_zn = info_zn['SettlingTime']
overshoot_zn = info_zn['Overshoot']

# Resultados de Ziegler-Nichols
print("\nResultados do Ziegler-Nichols:")
print("Tempo de Subida (ZN):", tempo_subida_zn)
print("Tempo de Acomodação (ZN):", tempo_acomodacao_zn)
print("Overshoot (ZN):", overshoot_zn)

# Cohen e Coon em malha aberta
Kp_cc = 0.9 / k
Ti_cc = 3 * theta
Td_cc = 0.33 * tau

# Criando o controlador PID com os parâmetros de Cohen e Coon
num_pid_cc = [Kp_cc * Td_cc, Kp_cc, Kp_cc / Ti_cc]
den_pid_cc = [1, 0]
PID_cc = ctrl.TransferFunction(num_pid_cc, den_pid_cc)

# Criando o sistema em série com os parâmetros de Cohen e Coon
Cs_cc = ctrl.series(PID_cc, sys_atraso)

# Gerando a resposta ao degrau do sistema em malha fechada com Cohen e Coon
tempo_resposta_cc, resposta_cc = ctrl.step_response(ctrl.feedback(Cs_cc, 1))

# Calculando informações adicionais usando a função step_info
info_cc = ctrl.step_info(ctrl.feedback(Cs_cc, 1))
tempo_subida_cc = info_cc['RiseTime']
tempo_acomodacao_cc = info_cc['SettlingTime']
overshoot_cc = info_cc['Overshoot']

# Resultados de Cohen e Coon
print("\nResultados de Cohen e Coon:")
print("Tempo de Subida (CC):", tempo_subida_cc)
print("Tempo de Acomodacao (CC):", tempo_acomodacao_cc)
print("Overshoot (CC):", overshoot_cc)

# Construindo a string da função de transferência com base nos parâmetros calculados para Ziegler-Nichols
num_str_zn = f'{Kp_zn * Td_zn:.4f}s^2 + {Kp_zn:.4f}s + {Kp_zn / Ti_zn:.4f}'
den_str_zn = f'{tau:.4f}s^2 + s'
Cs_str_zn = f'{num_str_zn}\n{"-" * 28}\n{" " * 7}{den_str_zn}'

# Construindo a string da função de transferência com base nos parâmetros calculados para Cohen e Coon
num_str_cc = f'{Kp_cc * Td_cc:.4f}s^2 + {Kp_cc:.4f}s + {Kp_cc / Ti_cc:.4f}'
den_str_cc = f'{tau:.4f}s^2 + s'
Cs_str_cc = f'{num_str_cc}\n{"-" * 28}\n{" " * 7}{den_str_cc}'

# Resultados das funções de transferência
print("\nFunção de transferência do sistema (ZN):\n")
print(Cs_str_zn)
print("\nKp (ZN):", Kp_zn)
print("Ti (ZN):", Ti_zn)
print("Td (ZN):", Td_zn)

print("\nFunção de transferência do sistema (CC):\n")
print(Cs_str_cc)
print("\nKp (CC):", Kp_cc)
print("Ti (CC):", Ti_cc)
print("Td (CC):", Td_cc)

# Plotando os resultados do Ziegler-Nichols
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(tempo_resposta_zn, resposta_zn, label='Ziegler-Nichols')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Ziegler-Nichols)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tempo, saida, 'r--')
plt.plot(tempo_resposta_zn, resposta_zn, 'b-', label='Ziegler-Nichols')
plt.xlabel('Tempo [s]')
plt.ylabel('Saída [°C]')
plt.title('Dados Reais vs Identificação (Ziegler-Nichols)')
plt.legend(['Identificação', 'Real'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Ziegler-Nichols.png'))

# Plotando os resultados de Cohen e Coon
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(tempo_resposta_cc, resposta_cc, label='Cohen e Coon')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Cohen e Coon)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tempo, saida, 'r--')
plt.plot(tempo_resposta_cc, resposta_cc, 'g-', label='Cohen e Coon')
plt.xlabel('Tempo [s]')
plt.ylabel('Saída [°C]')
plt.title('Dados Reais vs Identificação (Cohen e Coon)')
plt.legend(['Identificação', 'Real'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Cohen-e-Coon.png'))

# Perguntar ao usuário os parâmetros do PID e o Setpoint
Kp_user = float(input("\nDigite o valor de Kp para o novo PID: "))
Ti_user = float(input("Digite o valor de Ti para o novo PID: "))
Td_user = float(input("Digite o valor de Td para o novo PID: "))
Setpoint_user = float(input("Digite o valor do Setpoint: "))

# Criar o controlador PID com os novos parâmetros inseridos pelo usuário
num_pid_user = [Kp_user * Td_user, Kp_user, Kp_user / Ti_user]
den_pid_user = [1, 0]
PID_user = ctrl.TransferFunction(num_pid_user, den_pid_user)

# Criar o sistema em série com os novos parâmetros inseridos pelo usuário
Cs_user = ctrl.series(PID_user, sys_atraso)

# Gerar a resposta ao degrau do sistema em malha fechada com os novos parâmetros inseridos pelo usuário
tempo_resposta_user, resposta_user = ctrl.step_response(ctrl.feedback(Cs_user, 1))

# Plotar os resultados com os novos parâmetros inseridos pelo usuário
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(tempo_resposta_user, resposta_user, label='Parametros do Usuário')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Parametros do Usuário)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(tempo, saida, 'r--')
plt.plot(tempo_resposta_user, resposta_user, 'b-', label='Parametros do Usuário')
plt.xlabel('Tempo [s]')
plt.ylabel('Saída [°C]')
plt.title('Dados Reais vs Identificação (Parametros do Usuário)')
plt.legend(['Identificação', 'Real'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'ParametrosDoUsuário.png'))

plt.show()
