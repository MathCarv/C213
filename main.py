from scipy.io import loadmat
import numpy as np
import os
from scipy.signal import step, lti
import matplotlib.pyplot as plt
import control as ctrl

save_dir = 'graphics/'

# Carregar dados do arquivo .mat
mat = loadmat('dataset/Dataset_Grupo1.mat')
struct_degrau = mat.get('TARGET_DATA____ProjetoC213_Degrau')
degrau = struct_degrau[:, 1].tolist()  # vetor coluna
tempo = struct_degrau[:, 0].tolist()  # vetor coluna
struct_saida = mat.get('TARGET_DATA____ProjetoC213_Saida')
saida = struct_saida[:, 1].tolist()  # vetor coluna

AmplitudeDegrau = np.mean(degrau)
valorInicial = saida[0]

def Smith(Step, Tempo, Saída):
    if not isinstance(Tempo, list) or not isinstance(Saída, list) or len(Tempo) < 1 or len(Saída) < 1:
        raise TypeError('Erro de Tipo: Os argumentos devem ser listas não vazias.')
        
    if not isinstance(Step, (int, float)):
        raise TypeError('Erro de Tipo: O argumento \'Step\' deve ser uma constante.')
    
    Saída = [x - Saída[0] for x in Saída]
    valorFinal = Saída[-1]
    k = valorFinal / Step
    
    t1 = 0
    t2 = 0
    for i in range(len(Saída)):
        if Saída[i] >= 0.283 * valorFinal and t1 == 0:
            t1 = Tempo[i]
        
        if Saída[i] >= 0.6321 * valorFinal:
            t2 = Tempo[i]
            break
    
    tau = 1.5 * (t2 - t1)
    Theta = t2 - tau
    identificacaoSmith = [k, tau, Theta]
    
    return identificacaoSmith

def Sundaresan(Step, Tempo, Saída):
    if not isinstance(Tempo, list) or not isinstance(Saída, list) or len(Tempo) < 1 or len(Saída) < 1:
        raise TypeError('Erro de Tipo: Os argumentos devem ser listas não vazias.')
        
    if not isinstance(Step, (int, float)):
        raise TypeError('Erro de Tipo: O argumento \'Step\' deve ser uma constante.')
    
    Saída = [x - Saída[0] for x in Saída]
    valorFinal = Saída[-1]
    k = valorFinal / Step
    
    t1 = 0
    t2 = 0
    for i in range(len(Saída)):
        if Saída[i] >= 0.353 * valorFinal and t1 == 0:
            t1 = Tempo[i]
        
        if Saída[i] >= 0.853 * valorFinal:
            t2 = Tempo[i]
            break
    
    tau = 2 / 3 * (t2 - t1)
    Theta = 1.3 * t1 - 0.29 * t2
    identificacaoSundaresan = [k, tau, Theta]
    
    return identificacaoSundaresan

# Parametros da função Smith
smith = Smith(np.mean(degrau), tempo, saida)
sys_smith = ctrl.TransferFunction([smith[0], 1], [smith[1], 1])
t, y = ctrl.step_response(sys_smith * AmplitudeDegrau, tempo)
saida_smith = y +  valorInicial
erro_smith = np.sqrt(np.mean((saida - saida_smith)**2))

# Parametros da função Sundaresan
sundaresan = Sundaresan(np.mean(degrau), tempo, saida)
sys_sundaresan = ctrl.TransferFunction([sundaresan[0], 1], [sundaresan[1], 1])
t, y = ctrl.step_response(sys_sundaresan * AmplitudeDegrau, tempo)
saida_sundaresan = y +  valorInicial
erro_sundaresan = np.sqrt(np.mean((saida - saida_sundaresan)**2))

# Escolhendo através do erro
if erro_smith < erro_sundaresan:
    k = smith[0]
    tau = smith[1]
    theta = smith[2]
    print("O método escolhido foi smith!\n")

else :
    k = sundaresan[0]
    tau = sundaresan[1]
    theta = sundaresan[2]
    print("O método escolhido foi Sundaresan!\n")

print("Valor de K:", k)
print("Valor do atraso de transporte:", theta)
print("Valor da constante de tempo:", tau)

# Ziegler-Nichols em malha aberta
Kp_zn = (1.2 * tau) / (k * theta)
Ti_zn = 2 * theta
Td_zn = tau / 2

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
Kp_cc = (tau / (k * theta)) * ((16 * tau + 3 * theta) / (12 * tau))
Ti_cc = theta * (32 + (6 * theta) / tau) / (13 + (8 * theta) / tau)
Td_cc = (4 * theta) / (11 + (2 * theta / tau))

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

sys = (k, [tau, 1])
tout, yout = step(sys, T=np.linspace(0, tempo[-1], len(tempo)))

# Plotando o degrau de entrada e saída
plt.figure(figsize=(8, 6))
plt.plot(tempo, degrau, label='Degrau de Entrada')
plt.plot(tempo, saida, label='Degrau de Saída')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.title('Função de Transferência')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'FuncaoDeTransferencia.png'))

# Plotando os resultados do Ziegler-Nichols
plt.figure(figsize=(8, 6))
plt.plot(tempo_resposta_zn, resposta_zn, label='Ziegler-Nichols')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Ziegler-Nichols)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'RespostaAoDegrauZN.png'))

# Plotando os resultados de Cohen e Coon
plt.figure(figsize=(8, 6))
plt.plot(tempo_resposta_cc, resposta_cc, label='Cohen e Coon')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Cohen e Coon)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'RespostaAoDegrauCC.png'))

sys = (k, [tau, 1])
tout, yout = step(sys, T=np.linspace(0, tempo[-1], len(tempo)))

plt.figure(figsize=(8, 6))
plt.plot(tempo, saida, 'r--', label ='Real')
plt.plot(tout + theta, yout * AmplitudeDegrau, label='Identificação ')
plt.xlabel('Tempo [s]')
plt.ylabel('Saída')
plt.title('Dados Reais vs Identificação')
plt.legend(['Identificação', 'Real'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'Cohen-e-Coon.png'))

# Solicitar ao usuário os valores de k, tau, theta e setpoint
metodo = input("\nDigite o método desejado (Ziegler-Nichols (zn) ou Cohen e Coon (co)): ")
K_usuario = float(input("Digite o valor de K: "))
Tau_usuario = float(input("Digite o valor de tau: "))
Theta_usuario = float(input("Digite o valor de theta: "))
Setpoint_usuario = float(input("Digite o valor do setpoint: "))

# Calculando os parâmetros de acordo com o método escolhido
if metodo.lower() == 'zn':
    Kp_usuario = (1.2 * Tau_usuario) / (K_usuario * Theta_usuario)
    Ti_usuario = 2 * Theta_usuario
    Td_usuario = Tau_usuario / 2
elif metodo.lower() == 'co':
    Kp_usuario = (Tau_usuario / (K_usuario * Theta_usuario)) * ((16 * Tau_usuario + 3 * Theta_usuario) / (12 * Tau_usuario))
    Ti_usuario = Theta_usuario * (32 + (6 * Theta_usuario) / Tau_usuario) / (13 + (8 * Theta_usuario) / Tau_usuario)
    Td_usuario = (4 * Theta_usuario) / (11 + (2 * Theta_usuario / Tau_usuario))
else:
    print("Método inválido. Por favor, escolha entre Ziegler-Nichols e Cohen e Coon.")
    exit()

# Criar o controlador PID com os parâmetros calculados
num_pid_usuario = [Kp_usuario * Td_usuario, Kp_usuario, Kp_usuario / Ti_usuario]
den_pid_usuario = [1, 0]
PID_usuario = ctrl.TransferFunction(num_pid_usuario, den_pid_usuario)

# Criar o sistema em série com os parâmetros calculados
Cs_usuario = ctrl.series(PID_usuario, sys_atraso)

# Gerar a resposta ao degrau do sistema em malha fechada com os parâmetros calculados
tempo_resposta_usuario, resposta_usuario = ctrl.step_response(ctrl.feedback(Cs_usuario, 1))

# Função para calcular o erro quadrático médio
def calcular_erro_quadratico_medio(saida_real, saida_estimada):
    erro = np.sqrt(np.mean((saida_real - saida_estimada) ** 2))
    return erro

# Interpolação da resposta ao degrau do sistema para coincidir com os tempos dos dados reais
resposta_zn_interpolada = np.interp(tempo, tempo_resposta_zn, resposta_zn)
resposta_cc_interpolada = np.interp(tempo, tempo_resposta_cc, resposta_cc)
resposta_usuario_interpolada = np.interp(tempo, tempo_resposta_usuario, resposta_usuario)

# Calculando o erro para Ziegler-Nichols com os valores interpolados (malha fechada)
erro_zn = calcular_erro_quadratico_medio(saida, resposta_zn_interpolada)

# Calculando o erro para Cohen e Coon com os valores interpolados (malha fechada)
erro_cc = calcular_erro_quadratico_medio(saida, resposta_cc_interpolada)

# Calculando o erro para os parâmetros inseridos pelo usuário com os valores interpolados (malha fechada)
erro_usuario = calcular_erro_quadratico_medio(saida, resposta_usuario_interpolada)

# Calculando o erro para Ziegler-Nichols com os valores interpolados (malha aberta)
erro_zn_malha_aberta = calcular_erro_quadratico_medio(degrau, resposta_zn_interpolada)

# Calculando o erro para Cohen e Coon com os valores interpolados (malha aberta)
erro_cc_malha_aberta = calcular_erro_quadratico_medio(degrau, resposta_cc_interpolada)

# Calculando o erro para os parâmetros inseridos pelo usuário com os valores interpolados (malha aberta)
erro_usuario_malha_aberta = calcular_erro_quadratico_medio(degrau, resposta_usuario_interpolada)

# Imprimindo os erros
print("\nErros:")
print("Erro Ziegler-Nichols (malha fechada):", erro_zn)
print("Erro Cohen e Coon (malha fechada):", erro_cc)
print("Erro com parâmetros do usuário (malha fechada):", erro_usuario)
print("\nErro Ziegler-Nichols (malha aberta):", erro_zn_malha_aberta)
print("Erro Cohen e Coon (malha aberta):", erro_cc_malha_aberta)
print("Erro com parâmetros do usuário (malha aberta):", erro_usuario_malha_aberta)

# Plotar os resultados com os parâmetros inseridos pelo usuário
plt.figure(figsize=(12, 6))
plt.plot(tempo_resposta_usuario, resposta_usuario, label='Parametros do Usuário')
plt.xlabel('Tempo [s]')
plt.ylabel('Resposta ao Degrau')
plt.title('Resposta ao Degrau do Sistema em Malha Fechada (Parametros do Usuário)')
plt.legend(['Resposta ao Degrau'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'DegrauUsuario.png'))

sys2 = (K_usuario, [Tau_usuario, 1])
tout2, yout2 = step(sys2, T=np.linspace(0, tempo[-1], len(tempo)))
plt.figure(figsize=(12, 6))
plt.plot(tempo, saida, 'r--', label ='Real')
plt.plot(tout2 + Theta_usuario, yout2 * AmplitudeDegrau, label='Identificação')
plt.xlabel('Tempo [s]')
plt.ylabel('Saída')
plt.title('Dados Reais vs Identificação (Parametros do Usuário)')
plt.legend(['Identificação', 'Real'], loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'ParametrosDoUsuario.png'))

plt.show()
