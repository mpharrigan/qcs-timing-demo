from typing import List

import networkx as nx
import numpy as np

from pyquil.gates import H
from pyquil.paulis import exponential_map, sX
from pyquil.quil import Program, Pragma
from pyquil.quilatom import QubitPlaceholder


def generate_maxcut_ising(graph):
    """
    Creates a Maxcut problem where every edge is weight -1 and every node is offset 0

    :param graph:
    :return:
    """
    node_weights = {node: {'weight': 0} for node in graph.nodes}
    edge_weights = {edge: {'weight': -1} for edge in graph.edges}
    nx.set_node_attributes(graph, node_weights)
    nx.set_edge_attributes(graph, edge_weights)
    return graph


def _nx_to_node_array(graph, nodes, weight='weight'):
    return np.array([graph.nodes[n].get(weight, 0.0) for n in nodes])


def calculate_ising_rewards(graph: nx.Graph, bitstrings: np.ndarray):
    """
    Given an Ising problem encoded as a graph, return the reward values for an array of bitstrings.

    :param graph: A NetworkX graph where edges encode the interaction coefficients and nodes
        encode the one-body coefficients. The weights should be stored in attributes called
        'weight'. Nodes should be contiguous integers from 0 .. N
    :param bitstrings: An array of shape (n_bitstrings, n_bits) specifying the bitstrings
        for which we evaluate the reward values.
    :return: An array of reward values of shape (n_bitstrings,)
    """
    nodes = list(range(graph.number_of_nodes()))
    assert sorted(graph.nodes) == nodes, 'Problem must be described with contiguous integers'
    bitstrings = np.atleast_2d(bitstrings)
    twobody = nx.to_numpy_array(graph, nodes, weight='weight')
    onebody = _nx_to_node_array(graph, nodes, weight='weight')
    spinstrings = -2.0 * np.asarray(bitstrings) + 1.0

    # Apply to spin strings
    onebody = np.dot(spinstrings, onebody)
    twobody = np.sum(np.dot(spinstrings, twobody) * spinstrings, axis=-1)
    return 0.5 * twobody + onebody


def exponentiate_driver(pauli_sum, param):
    """
    Calculates the exponential representation of the given QAOA driver term

    :param pauli_sum: PauliSum of the driver
    :param param: parameter set for the driver
    :return: pyQuil.Program for the exponential
    """
    prog = Program()
    for term in pauli_sum.terms:
        prog += exponential_map(term)(param)
    return prog


def exponentiate_reward(pauli_sum, param):
    """
    Calculates the exponential representation of the given QAOA problem hamiltonian term. This is
    separated out from the driver as it allows for manual "compilation" by setting the edges and
    their order to reflect the topology of any hardware the code is supposed to run on.

    :param pauli_sum: PauliSum of the problem hamiltonian
    :param param: parameter set (angles) for the problem hamiltonian
    :param edge_ordering: List of pairs of connected qubits in the desired order, or None for
        the default order.
    :return: pyQuil.Program for the exponential
    """
    prog = Program()
    prog += Pragma('COMMUTING_BLOCKS')
    for term in pauli_sum.terms:
        prog += Pragma('BLOCK')
        prog += exponential_map(term)(param)
        prog += Pragma('END_BLOCK')
    prog += Pragma('END_COMMUTING_BLOCKS')
    return prog


def get_standard_driver(qubits: List[QubitPlaceholder]):
    return sum(sX(q) for q in qubits)


def get_qaoa_program(qubits, driver, reward, betas, gammas) -> Program:
    """
    Return the pyQuil program to construct the QAOA state for the problem given beta
    and gamma angles

    :param driver: The driver PauliSum. Usually X_i on all qubits.
    :param problem_ham: The PauliSum representing the problem reward
    :param betas: Beta angles for parameterizing the driver unitary.
    :param gammas: Gamma angles for parameterizing the problem hamiltonian unitary.
    :return: The program
    """
    assert len(betas) == len(gammas)
    prob_progs = [exponentiate_reward(reward, gamma) for gamma in gammas]
    driver_progs = [exponentiate_driver(driver, beta) for beta in betas]
    interleaved_progs = [prob_prog + driver_prog
                         for prob_prog, driver_prog in zip(prob_progs, driver_progs)]

    prog = Program([H(q) for q in qubits])
    for iprog in interleaved_progs:
        prog += iprog

    return prog
