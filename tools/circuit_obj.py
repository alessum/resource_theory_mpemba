import numpy as np
import functions as fn


class Circuit:
    def __init__(self, N, T, gates, order=None, symmetry=None, K=None):
        """
        Initialize the circuit object.
        - N is the number of qubits.
        - T is the number of time steps.
        - gates is either the list of gates or of parameters: 
            [circuit_type, geometry, Js] or [circuit_type, geometry]
            where if u1 is used, Js = [J, Jz] and if su2 is used, Js = J = Jz.
          if no Js is reported, the gates are set randomly.
        - initial_state is either the state or the parameters:
            [state_type, p, state_phases, theta]
        - order is the order of the gates.
        """
        self.N = N
        self.T = T
        self.order = order
        self.gates = gates
        self.symmetry = symmetry
        if symmetry == 'ZK':
            self.K = K
        else:
            self.K = 2
            
    def generate_unitary(self, masks_dict):
        """
        Generate the unitary operator for the circuit.
        """
        # apply the circuit to each state of the computational basis
        unitary = np.zeros((2**self.N, 2**self.N), dtype=np.complex128)
        for i in range(2**self.N):
            state = np.zeros((2**self.N,), dtype=np.complex128)
            state[i] = 1
            state = fn.apply_U(state, self.gates, self.order, masks_dict, (2 if self.symmetry!='ZK' else self.K))
            unitary[:, i] = state
        return unitary

    def run(self, masks_dict, sites_to_keep, alphas, rho, Ubasis=None):
        """
        Run the circuit and calculate the QFIs and EA.
        """
        assert len(rho.shape) == 2, "The input state must be a 2D array."
        sites_to_trace = [i for i in range(self.N) if i not in sites_to_keep]
        Ns = len(sites_to_keep)
        results = np.zeros((len(alphas), self.T + 1))
        rho_s_evolution = np.zeros((self.T + 1, 2**Ns, 2**Ns), dtype=np.complex128)
        
        U_op = self.generate_unitary(masks_dict)
        
        U_op      = np.asfortranarray(U_op)
        U_op_dag  = np.asfortranarray(U_op.conj().T)
                
        rho_s = fn.ptrace(rho, sites_to_keep) 
            
        if self.symmetry == 'U1':           
            rho_s_tw = fn.manual_U1_tw(rho_s, self.projectors)
        elif self.symmetry == 'SU2':
            rho_s_tw = fn.manual_N4_SU2_tw(rho_s)
        elif self.symmetry == 'Z2':
            rho_s_tw = fn.manual_Z2_tw(rho_s)
        elif self.symmetry == 'ZK':
            rho_s_tw = fn.manual_ZK_tw(rho_s, self.K)

        rho_e = fn.ptrace(rho, sites_to_trace)

        rho_s_evolution[0] = rho_s
        results[:, 0] = [fn.renyi_divergence(rho_s, rho_s_tw, alpha) for alpha in alphas]
        
        for t in range(1, self.T):
            rho_fr = np.asfortranarray(rho)
            rho = U_op @ rho_fr @ U_op_dag
            
            rho_s = fn.ptrace(rho, sites_to_keep) 
            
            if self.symmetry == 'U1':           
                rho_s_tw = fn.manual_U1_tw(rho_s, self.projectors)
            elif self.symmetry == 'SU2':
                rho_s_tw = fn.manual_N4_SU2_tw(rho_s)
            elif self.symmetry == 'Z2':
                rho_s_tw = fn.manual_Z2_tw(rho_s)
            elif self.symmetry == 'ZK':
                rho_s_tw = fn.manual_ZK_tw(rho_s, self.K)
            
            rho = np.kron(rho_e, rho_s) # <--- resetting the environment
            
            rho_s_evolution[t] = rho_s            
            results[:, t] = [fn.renyi_divergence(rho_s, rho_s_tw, alpha) for alpha in alphas]
                
        return results, rho_s_evolution

    def compute_hamiltonian(self, masks_dict):
        """
        Compute the Hamiltonian of the circuit.
        """
        return fn.compute_hamiltonian(self.gates, self.order, masks_dict, self.N)