"""
Scan over gamma for H = ZZ two-qubit QAOA

This does *not* use parametric binaries. It re-compiles each program.
"""
import os
import time

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pyquil import get_qc
from pyquil.api import QuantumComputer
from pyquil.gates import MEASURE
from pyquil.paulis import sX, sZ

from qaoa_utils import generate_maxcut_ising, calculate_ising_rewards, get_qaoa_program


def run_point(qc: QuantumComputer, beta: float, gamma: float, q0: int, q1: int,
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

    # Construct a program
    ising = generate_maxcut_ising(nx.from_edgelist([(0, 1)]))
    driver = sX(q0) + sX(q1)
    reward = sZ(q0) * sZ(q1) + 0  # add 0 to turn to PauliSum

    program = get_qaoa_program(qubits=[q0, q1], driver=driver, reward=reward,
                               betas=[beta], gammas=[gamma])
    ro = program.declare('ro', memory_size=2)
    program += MEASURE(q0, ro[0])
    program += MEASURE(q1, ro[1])
    program = program.wrap_in_numshots_loop(shots=n_shots)

    # Compile it
    nq_program = qc.compiler.quil_to_native_quil(program)
    executable = qc.compiler.native_quil_to_executable(nq_program)

    # Run and post-process
    bitstrings = qc.run(executable)
    rewards = calculate_ising_rewards(ising, bitstrings)
    return np.mean(rewards)


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
    start = time.time()
    for gamma in gammas:
        results.append({
            'beta': beta,
            'gamma': gamma,
            'expected_reward': run_point(qc, beta, gamma, q0, q1)
        })
    end = time.time()
    tot_time = end - start
    print(f"Total time: {tot_time:.2f}s")
    time_per_iter = tot_time / len(gammas)
    print(f"Time per iter: {time_per_iter*1000:.1f}ms")
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
    fn_base = f'non-parametric.{qc.name}.{q0}.{q1}'
    run_scan(fn=f'{fn_base}.json', qc=qc, q0=q0, q1=q1)

    qvm_fn = 'non-parametric.2q-qvm.0.1.json'
    if os.path.exists(qvm_fn):
        plot(qvm_fn, label='QVM')

    plot(f'{fn_base}.json', label='non-parametric')
    plt.xlabel('Gamma')
    plt.ylabel(r'$\langle \psi | ZZ | \psi \rangle$')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{fn_base}.png')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main(get_qc('Aspen-0-5Q-C'), q0=2, q1=15)
