
from cached_property import cached_property
from qiskit.opflow import CircuitStateFn, NaturalGradient
from qiskit import QuantumCircuit
from symmer import QuantumState, PauliwordOp
from scipy.optimize import minimize
import numpy as np
from typing import *
from copy import deepcopy

class VQE_Driver:
    # Other methods from the original class

    def f(self, x: np.array) -> float:
        """ 
        Given a parameter vector, bind to the circuit and retrieve expectation value.

        Args:
            x (np.array): Parameter vector

        Returns:
            Expectation Value (float)
        """
        if self.expectation_eval == 'observable_rotation':
            state = self.get_state(self.excitation_generators, x)
        else:
            state = self.get_state(self.circuit, x)
        return self._f(self.observable, state)

    def compute_fisher_information(self, x: np.array) -> np.array:
        """
        Compute the Fisher information matrix for the given parameter vector.

        Args:
            x (np.array): Parameter vector

        Returns:
            Fisher information matrix (np.array)
        """
        num_parameters = len(x)
        fisher_information = np.zeros((num_parameters, num_parameters))

        # Use Qiskit's Natural Gradient method to estimate the Fisher information matrix.
        nat_grad = NaturalGradient(regularization='ridge')
        bound_circuit = self.circuit.bind_parameters(x)
        operator = CircuitStateFn(self.observable @ bound_circuit)
        fisher_information = nat_grad.compute_fisher_information(operator, self.circuit.parameters)

        return np.asarray(fisher_information)

    def gradient(self, x: np.array) -> np.array:
        """
        Get the ansatz parameter gradient using Quantum Natural Gradient method.

        Args:
            x (np.array): Parameter vector

        Returns:
            Ansatz parameter natural gradient (np.array)
        """
        # Compute the Fisher information matrix
        fisher_information_matrix = self.compute_fisher_information(x)

        # Compute the gradient
        grad = np.zeros(self.circuit.num_parameters)

        for i in range(self.circuit.num_parameters):
            grad[i] = self.evaluate_partial_derivative(x, i)

        # Compute the natural gradient using the inverse of the Fisher information matrix
        natural_gradient = np.linalg.inv(fisher_information_matrix).dot(grad)

        return np.asarray(natural_gradient)

    def evaluate_partial_derivative(self, x: np.array, param_index: int, delta=1e-8) -> float:
        """
        Calculate the partial derivative of the expectation value with respect to a parameter.

        Args:
            x (np.array): Parameter vector
            param_index (int): Index of the parameter to differentiate
            delta (float): Small change for numerical differentiation

        Returns:
            Partial derivative (float)
        """
        x_plus = np.copy(x)
        x_minus = np.copy(x)

        # Add and subtract small delta to compute the finite difference
        x_plus[param_index] += delta
        x_minus[param_index] -= delta

        # Compute expectation values
        f_plus = self.f(x_plus)
        f_minus = self.f(x_minus)

        # Central difference method for partial derivative
        return (f_plus - f_minus) / (2 * delta)

class ADAPT_VQE(VQE_Driver):
    def optimize(self, 
            max_cycles:int=10, gtol:float=1e-3, atol:float=1e-10, 
            target:float=0, target_error:float=1e-3
        ):
        """ 
        Perform the ADAPT-VQE optimization

        Args:
            gtol: gradient threshold below which optimization will terminate
            atol: if the difference between successive expectation values is below this threshold, terminate
            max_cycles: maximum number of ADAPT cycles to perform
            target: if a target energy is known, this may be specified here
            target_error: the absoluate error threshold with respect to the target energy 
        """
        interim_data = {'history':[]}
        adapt_cycle=1
        gmax=1
        anew=1
        aold=0
        
        while (
                gmax>gtol and adapt_cycle<=max_cycles and 
                abs(anew-aold)>atol and abs(anew-target)>target_error
            ):
            # save the previous gmax to compare for the gdiff check
            aold = deepcopy(anew)
            # calculate gradient across the pool and select term with the largest derivative
            scores = self.pool_score()
            grad_rank = list(map(int, np.argsort(scores)[::-1]))
            gmax = scores[grad_rank[0]]

            # TETRIS-ADAPT-VQE
            if self.TETRIS:
                new_excitation_list = []
                support_mask = np.zeros(self.observable.n_qubits, dtype=bool)
                for i in grad_rank:
                    new_excitation = self.excitation_pool[i]
                    support_exists = (new_excitation.X_block | new_excitation.Z_block) & support_mask
                    if ~np.any(support_exists):
                        new_excitation_list.append(new_excitation)
                        support_mask = support_mask | (new_excitation.X_block | new_excitation.Z_block)
                    if np.all(support_mask) or scores[i] < gtol:
                        break
            else:
                new_excitation_list = [self.excitation_pool[grad_rank[0]]]
                
            # append new term(s) to the adapt_operator that stores our ansatz as it expands
            n_new_terms = len(new_excitation_list)
            self.append_to_adapt_operator(new_excitation_list)
                        
            if self.verbose:
                print('-'*39)
                print(f'ADAPT cycle {adapt_cycle}
')
                print(f'Largest pool derivative ∂P∂θ = {gmax: .5f}
')
                print('Selected excitation generator(s):
')
                for op in new_excitation_list:
                    print(f'	{symplectic_to_string(op.symp_matrix[0])}')
                print('
', '-'*39)
            
            # Replacing BFGS optimization with natural gradient optimization
            self.prepare_for_evolution(self.adapt_operator)
            
            opt_out = self.natural_gradient_optimization(
                x0=np.append(self.opt_parameters, [0]*n_new_terms), max_steps=100
            )
            
            anew = opt_out['fun']
            interim_data[adapt_cycle] = {
                'output':opt_out, 'gmax':gmax, 
                'excitation': [symplectic_to_string(t.symp_matrix[0]) for t in new_excitation_list]
            }
            interim_data['history'].append(anew)
            
            if self.verbose:
                print(F'
Energy at ADAPT cycle {adapt_cycle}: {anew: .5f}
')
            
            self.opt_parameters = opt_out['x']
            adapt_cycle+=1

        return {
            'result': opt_out, 
            'interim_data': interim_data,
            'ref_state': safe_QuantumState_to_dict(self.ref_state),
            'adapt_operator': [symplectic_to_string(t) for t in self.adapt_operator.symp_matrix]
        }

    def natural_gradient_optimization(self, x0, max_steps=100, learning_rate=0.1):
        """
        Perform natural gradient optimization for the given parameters.
        
        Args:
            x0 (np.array): Initial parameter vector.
            max_steps (int): Maximum number of optimization steps.
            learning_rate (float): Step size for gradient descent.
        
        Returns:
            opt_out (dict): Optimized parameters and function value.
        """
        x = x0.copy()
        history = {'params': [], 'energy': []}

        for step in range(max_steps):
            # Compute the natural gradient
            grad = self.gradient(x)
            
            # Update the parameters using gradient descent
            x -= learning_rate * grad

            # Evaluate the function value (energy)
            energy = self.f(x)

            history['params'].append(x.copy())
            history['energy'].append(energy)

            if self.verbose:
                print(f'Step {step}, Energy: {energy:.5f}')

            # Check for convergence (optional, based on energy difference or gradient norm)
            if step > 1 and abs(history['energy'][-1] - history['energy'][-2]) < 1e-6:
                break

        return {'x': x, 'fun': energy}
