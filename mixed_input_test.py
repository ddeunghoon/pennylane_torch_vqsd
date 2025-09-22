import pennylane as qml
from pennylane import numpy as np
import torch
from shc_povm import *


if __name__ == "__main__":

    def prepare_entangled_state(fixed_params):
        """2큐비트 순수 상태를 만드는 회로"""
        qml.RY(fixed_params[0], wires=0)  # 시스템 큐비트
        qml.CNOT(wires=[0, 1])           # 시스템-보조 큐비트 얽힘

    dev_cpu = qml.device("default.qubit", wires=2)

    @qml.qnode(dev_cpu)
    def create_density_matrix_A():
        fixed_params = np.array([np.pi / 5], requires_grad=False)
        prepare_entangled_state(fixed_params)
        
        return qml.density_matrix(wires=1)

    rho_A = create_density_matrix_A()

    probabilities, pure_states = np.linalg.eigh(rho_A)

    # PennyLane은 열벡터를 사용하므로, 고유벡터 행렬을 전치(transpose)
    pure_states = pure_states.T

    dev_gpu = qml.device("lightning.gpu", wires=1)
    @qml.qnode(dev_gpu)
    def variational_circuit_on_gpu(var_params, pure_state_input):
        """하나의 순수 상태를 입력받아 변분 연산을 수행"""
        # qml.StatePrep으로 입력 상태를 |psi_i>로 초기화
        qml.StatePrep(pure_state_input, wires=0)
        
        # 변분 회로
        qml.RX(var_params[0], wires=0)
        qml.RY(var_params[1], wires=0)
        
        # 측정
        return qml.expval(qml.PauliZ(0))

    # 3-2. 변분 회로의 학습 파라미터 초기화
    var_params = np.array([0.54, 0.12], requires_grad=True)

    # 3-3. 각 순수 상태에 대해 변분 회로를 실행하고 결과를 저장
    results_per_state = []
    for p, state in zip(probabilities, pure_states):
        # 확률이 0에 가까우면 계산할 필요 없음
        if not np.isclose(p, 0):
            # GPU QNode 실행
            exp_val = variational_circuit_on_gpu(var_params, pure_state_input=state)
            results_per_state.append(exp_val)
            print(f"p={p:.4f}인 |psi> 입력 시 결과: {exp_val:.4f}")

    # 3-4. 최종 기댓값 계산 (확률 가중 평균)
    # np.dot은 벡터 내적을 수행: (p1*res1 + p2*res2 + ...)
    final_expectation = np.dot(probabilities, np.array(results_per_state))
