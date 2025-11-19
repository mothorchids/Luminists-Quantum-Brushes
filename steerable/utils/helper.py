'''
Author: Chih-Kang Huang && chih-kang.huang@hotmail.com
Date: 2025-11-13 08:14:18
LastEditors: Chih-Kang Huang && chih-kang.huang@hotmail.com
LastEditTime: 2025-11-16 18:25:07
FilePath: /steerable/utils/helper.py
Description: 


'''
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx
import pennylane as qml
import matplotlib.pyplot as plt

"""Define Circuit with 2nd order trotterization"""
def splitting_circuit(model, initial_state, H_list, n_qubits, T=1.0, n_steps=40, n= 1):
    """ 
    model: control NN
    initial_state: Initial Quantum State
    H_list: Hamiltanian list
    T: final time
    n_steps: time steps
    n: trotterizaiton order
    """
    dt = T / n_steps
    qml.StatePrep(initial_state, wires=range(n_qubits))
    for k in range(n_steps):
        t_k = k * dt
        u_k = model(jnp.array(t_k))
        # Strang-splitting time step
        H0 = H_list[0]
        qml.ApproxTimeEvolution(H0, dt/2, 1)
        for u, H in zip(list(u_k), H_list[1:]): 
            qml.ApproxTimeEvolution(u*H, dt/2, 1)
        for u, H in (zip(reversed(list(u_k)), reversed(H_list[1:]))): 
            qml.ApproxTimeEvolution(u*H, dt/2, 1)
        qml.ApproxTimeEvolution(H0, dt/2, 1)
    return qml.state()

def density_matrix(psi):
    """Convert a state vector to a density matrix."""
    return jnp.outer(psi, jnp.conjugate(psi))

def quantum_fidelity(psi, rho):
    psi = psi/jnp.linalg.norm(psi)
    rho = rho /jnp.linalg.norm(rho)
    return jnp.abs(jnp.vdot(psi, rho))**2


### Visualization
def von_neumann_entropy(rho):
    """Von Neumann entropy in bits."""
    eigvals = jnp.real(jnp.linalg.eigvals(rho))
    eigvals = jnp.clip(eigvals, 1e-12, 1.0)
    return -jnp.sum(eigvals * jnp.log2(eigvals))

X = jnp.array(qml.matrix(qml.PauliX(0)))
Y = jnp.array(qml.matrix(qml.PauliY(0)))
Z = jnp.array(qml.matrix(qml.PauliZ(0)))
I = jnp.eye(2)

def bloch_vector(rho):
    """Compute Bloch vector for single-qubit density matrix."""
    return jnp.array([
        jnp.real(jnp.trace(rho @ X)),
        jnp.real(jnp.trace(rho @ Y)),
        jnp.real(jnp.trace(rho @ Z))
    ])

def partial_trace(psi, keep, n_qubits):
    """Partial trace over all qubits except those in 'keep' (list of indices)."""
    rho = jnp.outer(psi, jnp.conjugate(psi))
    dims = [2] * n_qubits
    rho = rho.reshape(dims + dims)

    # Trace out all qubits not in 'keep'
    for q in reversed(range(n_qubits)):
        if q not in keep:
            rho = jnp.trace(rho, axis1=q, axis2=q + n_qubits)
            n_qubits -= 1
    return rho

# ---------- Main visualization ----------

def visualize_bloch_trajectories(states, target_state, n_qubits, ent=False, savepath=None):
    """
    states: list/array of shape (T, 2**n)
    target_state: vector of shape (2**n,)
    n_qubits: int
    """

    # Compute single-qubit trajectories
    trajs = []
    targets = []
    for q in range(n_qubits):
        traj_q = jnp.array([
            bloch_vector(partial_trace(psi, [q], n_qubits)) for psi in states
        ])
        trajs.append(traj_q)
        targets.append(bloch_vector(partial_trace(target_state, [q], n_qubits)))


    # ---------- Visualization ----------
    fig = plt.figure(figsize=(5 * n_qubits, 5))
    n_fig = n_qubits +1 if ent else n_qubits
    # Each qubit's Bloch trajectory
    for q in range(n_qubits):
        ax = fig.add_subplot(1, n_fig, q + 1, projection='3d')
        traj = trajs[q]
        target = targets[q]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], lw=2)
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], color='green', label='start')
        ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], color='red', label='end')
        ax.scatter(target[0], target[1], target[2], color='blue', label='target')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'Qubit {q} Bloch trajectory')
        ax.legend()
    # Entanglement entropy
    if ent:
        # Entanglement entropy between qubit 0 and the rest
        ent_entropy = jnp.array([
            von_neumann_entropy(partial_trace(psi, [0], n_qubits)) for psi in states
        ])
        ax_e = fig.add_subplot(1, n_fig, n_fig)
        ax_e.plot(jnp.linspace(0, 1.0, len(ent_entropy)), ent_entropy, color='purple', lw=2)
        ax_e.set_xlabel('Time t')
        ax_e.set_ylabel('Entanglement entropy S(t)')
        ax_e.set_title('Entanglement entropy (qubit 0 vs rest)')
        ax_e.grid(True)

    plt.tight_layout()
    if savepath: 
        plt.savefig(savepath)
    else:
        plt.show()

def build_hamiltonians(n_qubits): 
    H0 = sum(qml.PauliX(i) @ qml.PauliX(i+1) for i in range(n_qubits-1))   
    H0 += sum(qml.PauliY(i) @ qml.PauliY(i+1) for i in range(n_qubits-1))  
    H0 += sum(qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits-1))

    if n_qubits == 1: 
        H_list  = [
            qml.PauliZ(0),
            qml.PauliX(0)
        ]
    elif n_qubits == 2:
        H_list = [
            H0, 
            qml.PauliX(0) @ qml.Identity(1),
            qml.Identity(0) @ qml.PauliY(1),
        ]
    elif n_qubits == 3: 
        J = jnp.array([0.2, 0.13])
        H_list = [
            H0,
            qml.PauliX(0) @ qml.Identity(1) @ qml.Identity(2),
            qml.Identity(0) @ qml.PauliY(1) @ qml.Identity(2),
            qml.Identity(0) @ qml.Identity(1) @ qml.PauliX(2),
        ]
    elif n_qubits == 4: 
        H_list = [
            H0,
            qml.PauliX(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.PauliY(1) @ qml.Identity(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.Identity(1) @ qml.PauliZ(2) @ qml.Identity(3),
            qml.Identity(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.PauliX(3),
#            qml.PauliZ(0) @ qml.Identity(1) @ qml.Identity(2) @ qml.Identity(3) ,
#            qml.Identity(0) @ qml.PauliZ(1) @ qml.Identity(2) @ qml.Identity(3) ,
        ]
    else:
        raise AssertionError("Not implemented yet")
        
    return H_list