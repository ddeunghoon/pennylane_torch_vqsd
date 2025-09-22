import pennylane as qml
from pennylane import numpy as np
from functools import partial
import torch
import torch.nn as nn
import pandas as pd
import math

from dataclasses import dataclass
from typing import Optional, Tuple, List

def rdm(state):
    """
    Statevector를 밀도 행렬로 변환하는 일반화된 함수.
    Args:
        state (np.array): 2^n 차원의 statevector.
    Returns:
        np.array: 2^n x 2^n 차원의 밀도 행렬.
    """
    return np.outer(state, state.conjugate().T)

def tp_maximization_cost(probs_povm, n, a_priori_probs):
    q = a_priori_probs
    res = 1
    for i in range(n):
        res -= q[i] * probs_povm[i][i]
    return res

def gray_code_bits(nbits: int, msb_first: bool = True):
    """
    nbits-비트 Gray code를 비트튜플의 리스트로 반환.
    msb_first=True면 (MSB,...,LSB) 순서의 튜플을 돌려줌.
    예: nbits=2 -> [(0,0), (0,1), (1,1), (1,0)]
    """
    seq = []
    for i in range(1 << nbits):
        g = i ^ (i >> 1)  # Gray code 정수
        if msb_first:
            bits = tuple((g >> (nbits - 1 - j)) & 1 for j in range(nbits))
        else:
            bits = tuple((g >> j) & 1 for j in range(nbits))
        seq.append(bits)
    return seq


def gray_code_toggle_indices(bits_seq):
    """
    연속한 Gray code 상태들 사이에서 '바뀐 비트의 인덱스' 목록을 반환.
    (각 단계에서 정확히 한 비트만 바뀌므로 인덱스는 1개씩)
    """
    toggles = []
    for prev, curr in zip(bits_seq[:-1], bits_seq[1:]):
        idx = next(i for i, (a, b) in enumerate(zip(prev, curr)) if a != b)
        toggles.append(idx)
    return toggles


@dataclass
class CSDNode:
    # 중앙 블록 파라미터(길이 = 2^(k), k = len(sys_wires)-1)
    alpha: Optional[np.ndarray] = None  # UCRz 앞
    beta:  Optional[np.ndarray] = None  # UCRy (== cs_paras)
    gamma: Optional[np.ndarray] = None  # UCRz 뒤

    # 재귀 서브블록 (한 큐빗 줄어든 문제; wires에서 target 제외한 부분에 적용)
    L0: Optional["CSDNode"] = None
    L1: Optional["CSDNode"] = None
    R0: Optional["CSDNode"] = None
    R1: Optional["CSDNode"] = None

    # 리프(단일 큐빗)일 때: 임의 U(2) -> ZYZ 형태(qml.Rot(phi, theta, omega))
    leaf_rot: Optional[Tuple[float, float, float]] = None


class POVM_Torch_Trainer():
    def __init__(self, n, a_priori_probs, optimizer, model, criterion):
        '''
        outcome 별로 서로 다른 dev에 매칭해서 기댓값을 뽑아냄
        __call__(): for inference
        evaluate(): for evaluating the performance 
        '''
        self.n_outcome = n
        # Prior Probabilities for each state
        self.a_priori_probs = a_priori_probs

        self.opt = optimizer
        self.model = model
        self.criterion = criterion

        self.verbose = 0
        self._eval_interval = 20
        self.history_ = []  # list of dicts: {'step': k, 'train_loss': ..., 'val_loss': ...}

    @torch.no_grad()
    def _log(self, step, loss):
        if self.verbose and step % self._eval_interval == 0:
            print(f"step {step:05d} | loss {loss:.7f}")

    def fit(self, X, steps=1000, verbose: int = 1, eval_interval: int = 20):
        # TODO: _make_optimizer()로 다양한 opt 실험
        self.history_.clear()
        self.verbose = verbose
        self._eval_interval = eval_interval

        P = self.model(X) ## index
        loss = self.criterion(P, self.n_outcome, self.a_priori_probs)
        loss_val = float(loss.detach().cpu())
        self.history_.append({"step": 0, "train_loss": loss_val})
        if self.verbose:
            msg = f"Cost(init_params): {loss:.7f}"
            print(msg)
        
        prev = None
        for step in range(1, steps + 1):
            self.opt.zero_grad()
            P = self.model(X)
            print(P)
            loss = self.criterion(P, self.n_outcome, self.a_priori_probs)
            loss.backward()
            self.opt.step()

            loss_val = float(loss.detach().cpu())
            # TODO: validatation_cost_fn을 입력 받아서 history에 val_loss 기록
            self.history_.append({"step": step, "train_loss": loss_val})
            self._log(step, loss_val)
            
            if prev is not None and abs(prev - loss_val) < 1e-7:
                break
            prev = loss_val

        return self.history_


class SingleQubitPOVM(nn.Module):
    # POVM PQC model on single system qubit
    def __init__(self, n_prep, n_sys, n_outcome, cfg, dev=None):
        super().__init__()
        self.n_outcome = n_outcome
        self.n_sys = n_sys
        n_anc = math.ceil(math.log2(n_outcome))
        total = n_prep + n_anc

        self.all_wires = list(range(total))
        self.prep_wires = list(range(n_prep))
        self.sys_wires = list(range(n_prep-n_sys, n_prep))
        self.povm_wires = list(range(n_prep-n_sys, total))
        self.anc_wires = list(range(n_prep, total))

        self.dev = qml.device("lightning.qubit", wires=self.all_wires)

        # num_params = (2**(2*n_sys)-1+2**n_sys) * (n_outcome-1)    # SU(3)
        num_params = 2**(n_sys-1)   # 3qubit unitary
        self.block_thetas = nn.ParameterList()
        param = nn.Parameter(
            torch.empty(num_params, dtype=cfg["dtype"], device=cfg["device"])
        )
        with torch.no_grad():
            param.uniform_(0.0, 2 * math.pi)

        self.block_thetas.append(param)
    
    def cs_block(self, sys_wires, cs_paras, msb_first=True):
        '''
        For n-qubit unitary, CS block requires 2^(n-1) numbers of rotation angle.
        This is just small test for 3qubit csd cs-block
        TODO: from UCRy_2controls to UCRy_(n-1)controls
        '''
        controls = sys_wires[:-1]
        targ = sys_wires[-1]
        k = len(controls)

        bits_seq = gray_code_bits(k, msb_first=msb_first)
        toggles  = gray_code_toggle_indices(bits_seq)
        restore_idx = 0 if msb_first else (k - 1)

        qml.RY(cs_paras[0], wires=targ)
        for i, t_idx in enumerate(toggles, start=1):
            qml.CNOT(wires=[controls[t_idx], targ])
            qml.RY(cs_paras[i], wires=targ)
        qml.CNOT(wires=[controls[restore_idx], targ])
    
    def ucrz_block(self, sys_wires, angles, msb_first=True):
        """
        UCRz(angles): ucry와 동일한 그레이코드 패턴으로 RZ를 적용
        angles 길이는 2^(k)
        """
        controls = sys_wires[:-1]
        targ = sys_wires[-1]
        k = len(controls)

        bits_seq = gray_code_bits(k, msb_first=msb_first)
        toggles  = gray_code_toggle_indices(bits_seq)
        restore_idx = 0 if msb_first else (k - 1)

        qml.RZ(angles[0], wires=targ)
        for i, t_idx in enumerate(toggles, start=1):
            qml.CNOT(wires=[controls[t_idx], targ])
            qml.RZ(angles[i], wires=targ)
        qml.CNOT(wires=[controls[restore_idx], targ])

    def apply_blockdiag_subtree(self, node0: Optional[CSDNode], node1: Optional[CSDNode], ctrl_wire, sub_wires):
        """
        (U0 ⊕ U1)을 회로로: 타깃(=ctrl_wire)이 |0>일 때 node0, |1>일 때 node1를 sub_wires에 적용
        """
        if node0 is not None:
            qml.ctrl(self.apply_csd_subtree, control=ctrl_wire, control_values=0)(node0, sub_wires)
        if node1 is not None:
            qml.ctrl(self.apply_csd_subtree, control=ctrl_wire, control_values=1)(node1, sub_wires)

    def apply_csd_subtree(self, node: CSDNode, sys_wires: List[int], msb_first=True):
        """
        sys_wires = [controls..., target] (마지막이 중앙블록의 target qubit)
        """
        if len(sys_wires) == 1:
            # 리프: 단일 큐빗 U(2)
            phi, theta, omega = node.leaf_rot
            qml.Rot(phi, theta, omega, wires=sys_wires[0])
            return

        controls = sys_wires[:-1]
        targ = sys_wires[-1]

        # 1) 오른쪽 블록부터 적용 (행렬곱의 오른쪽이 먼저)
        self.apply_blockdiag_subtree(node.R0, node.R1, targ, controls)

        # 2) 중앙 CS 블록: UCRz(alpha) -> UCRy(beta) -> UCRz(gamma)
        k = len(controls)
        if node.alpha is not None:
            assert len(node.alpha) == (1 << k)
            self.ucrz_block(sys_wires, node.alpha, msb_first=msb_first)
        if node.beta is not None:
            assert len(node.beta) == (1 << k)
            self.ucry_block(sys_wires, node.beta, msb_first=msb_first)
        if node.gamma is not None:
            assert len(node.gamma) == (1 << k)
            self.ucrz_block(sys_wires, node.gamma, msb_first=msb_first)

        # 3) 왼쪽 블록(마지막에 적용)
        self.apply_blockdiag_subtree(node.L0, node.L1, targ, controls)

    

    # def vqsd_fn(self):
    #     qml.Rot(self.theta[0], self.theta[1], self.theta[2], wires=self.sys_wires)

    #     # Controlled-RY gate controlled by first qubit in |0> state
    #     qml.PauliX(wires=self.sys_wires)
    #     qml.CRY(self.theta[3], wires=self.povm_wires)
    #     qml.PauliX(wires=self.sys_wires)
        
    #     # Controlled-RY gate controlled by first qubit in |1> state
    #     qml.CRY(self.theta[4], wires=self.povm_wires)

    # def _circuit_template(self, prep_fn):
    #     prep_fn(self.prep_wires)
    #     self.vqsd_fn()
    #     return qml.probs(wires=self.anc_wires)
    
    def _make_tape(self, prep_fn):
        """prep_fn(wires)와 HEAD를 결합한 단일 Tape 생성"""
        with qml.tape.QuantumTape() as tape:
            # 입력 상태 준비
            prep_fn(self.prep_wires)
            # 공통 HEAD + 확률 측정
            # self.vqsd_fn()
            self.cs_block(self.sys_wires, self.block_thetas[0])
            qml.probs(wires=self.anc_wires)
        return tape
    
    def forward(self, prep_batch):
        tapes = [self._make_tape(prep_fn) for prep_fn in prep_batch]

        results = qml.execute(
            tapes, 
            self.dev, 
            diff_method="parameter-shift", 
            interface="torch"
        )

        return torch.stack(results)
    
    # def _total_circ(self):
    #     self.prep_fn(self.prep_wires)
    #     self.vqsd_fn()
    #     return qml.probs(wires=self.anc_wires)

    # def _get_qnode(self, prep_fn):
    #     self.prep_fn = prep_fn
    #     return qml.QNode(func=self._total_circ, device=self.dev, interface="torch", diff_method="adjoint")


    # def forward(self, prep_batch):
    #     outs = []
    #     for prep_fn in prep_batch:
    #         q = self._get_qnode(prep_fn)
    #         outs.append(q())
    #     return torch.stack(outs)