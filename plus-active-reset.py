"""
Scan over gamma for H = ZZ two-qubit QAOA

This uses parametric binaries. It re-compiles each program.
"""
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyquil import get_qc, Program
from pyquil.api import QuantumComputer
from pyquil.gates import MEASURE, RESET
from pyquil.paulis import sX, sZ

from qaoa_utils import generate_maxcut_ising, calculate_ising_rewards, get_qaoa_program


def get_parametric_program(qc: QuantumComputer, q0: int, q1: int,
                           n_shots: int = 1000):
    """
    Run one (beta, gamma) point for the simplest Maxcut QAOA.

    :param qc: The QuantumComputer to use
    :param beta: The beta angle (p = 1)
    :param gamma: The gamma angle (p = 1)
    :param q0: The index of the first qubit
    :param q1: The index of the second qubit
    :param n_shots: The number of shots to take for this (beta, gamma) point.
    """
    ising = generate_maxcut_ising(nx.from_edgelist([(0, 1)]))
    driver = sX(q0) + sX(q1)
    reward = sZ(q0) * sZ(q1) + 0  # add 0 to turn to PauliSum

    program = Program(RESET())
    beta = program.declare('beta', memory_type='REAL')
    gamma = program.declare('gamma', memory_type='REAL')
    program += get_qaoa_program(qubits=[q0, q1], driver=driver, reward=reward,
                                betas=[beta], gammas=[gamma])
    ro = program.declare('ro', memory_size=2)
    program += MEASURE(q0, ro[0])
    program += MEASURE(q1, ro[1])
    program = program.wrap_in_numshots_loop(shots=n_shots)

    executable = qc.compile(program)
    return executable, ising


def run_scan(fn: str, qc: QuantumComputer, q0: int, q1: int):
    """Run a scan over gamma for a simple ZZ Hamiltonian.

    :param fn: Filename to save results
    :param qc: QuantumComputer to use
    :param q0: The index of the first qubit
    :param q1: The index of the second qubit
    """
    results = []
    gammas = np.linspace(0, np.pi, 50)
    beta = np.pi / 8
    executable, ising = get_parametric_program(qc, q0, q1)

    start = time.time()
    for gamma in gammas:
        bitstrings = qc.run(executable, memory_map={
            'beta': [gamma],
            'gamma': [gamma]
        })
        rewards = calculate_ising_rewards(ising, bitstrings)
        exp_reward = np.mean(rewards)
        results.append({
            'beta': beta,
            'gamma': gamma,
            'expected_reward': exp_reward,
        })
    end = time.time()
    tot_time = end - start
    print(f"Total time: {tot_time:.2f}s")
    time_per_iter = tot_time / len(gammas)
    print(f"Time per iter: {time_per_iter * 1000:.1f}ms")
    df = pd.DataFrame(results)
    df.to_json(fn)


def plot(fn, label=None):
    """Plot the results for a scan over gamma.

    :param fn: Filename to load results
    :param label: An optional label for the line on the plot
    """
    df = pd.read_json(fn).sort_values(by='gamma')
    plt.plot(df['gamma'], df['expected_reward'], '.-', label=label)


def main(qc, q0, q1):
    fn_base = f'reset.{qc.name}.{q0}.{q1}'
    run_scan(fn=f'{fn_base}.json', qc=qc, q0=q0, q1=q1)

    qvm_fn = 'non-parametric.2q-qvm.0.1.json'
    if os.path.exists(qvm_fn):
        plot(qvm_fn, label='QVM')

    non_param_fn = f'non-parametric.{qc.name}.{q0}.{q1}.json'
    if os.path.exists(non_param_fn):
        plot(non_param_fn, label='non-parametric')

    non_param_fn = f'parametric.{qc.name}.{q0}.{q1}.json'
    if os.path.exists(non_param_fn):
        plot(non_param_fn, label='parametric')

    plot(f'{fn_base}.json', label='reset')
    plt.xlabel('Gamma')
    plt.ylabel(r'$\langle \psi | ZZ | \psi \rangle$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{fn_base}.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main(get_qc('Aspen-1-16Q-A'), q0=13, q1=14)
