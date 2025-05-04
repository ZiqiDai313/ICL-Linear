import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch.nn as nn
from pyhessian import hessian
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from copy import deepcopy

class ModelWrapper(nn.Module):
    def __init__(self, model, output_dim):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.output_dim = output_dim
        
    def forward(self, x):
        # Ensure input has correct dimensions [batch_size, seq_len, dim]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        output = self.model(x)
        return output[:, -1, :self.output_dim]  # Return only the relevant output

class HessianAnalyzer:
    """Analyzes the Hessian of a model's loss function."""
    
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
        self.hessian_comp = None
        
    def compute_hessian_trace(self, inputs, targets):
        """Compute Hessian trace using PyHessian."""
        # Create a wrapper model that handles the specific input format
        model_wrapper = ModelWrapper(self.model, targets.shape[-1])
        model_wrapper = model_wrapper.to(self.device)
        model_wrapper.eval()
            
        if self.hessian_comp is None:
            self.hessian_comp = hessian(model_wrapper, self.criterion, 
                                    data=(inputs.to(self.device), targets.to(self.device)),
                                    cuda=torch.cuda.is_available())
        
        # Get trace and convert to float
        trace = self.hessian_comp.trace()
        if isinstance(trace, (list, tuple)):
            trace = float(np.mean(trace))
        elif isinstance(trace, torch.Tensor):
            trace = float(trace.item())
        else:
            trace = float(trace)
            
        return trace
        
    def compute_eigenvalues(self, inputs, targets, top_n=5):
        """Compute top eigenvalues of Hessian using PyHessian."""
        # Create a wrapper model that handles the specific input format
        model_wrapper = ModelWrapper(self.model, targets.shape[-1])
        model_wrapper = model_wrapper.to(self.device)
        model_wrapper.eval()
            
        if self.hessian_comp is None:
            self.hessian_comp = hessian(model_wrapper, self.criterion, 
                                    data=(inputs.to(self.device), targets.to(self.device)),
                                    cuda=torch.cuda.is_available())
        
        # Get eigenvalues and convert to list of floats
        eigenvalues = self.hessian_comp.eigenvalues(top_n=top_n)
        if isinstance(eigenvalues, tuple):
            eigenvalues = eigenvalues[0]  # Take only eigenvalues, not eigenvectors
        
        # Convert eigenvalues to list of floats
        if isinstance(eigenvalues, torch.Tensor):
            eigenvalues = eigenvalues.tolist()
        elif isinstance(eigenvalues, list):
            eigenvalues = [float(e) if isinstance(e, (int, float)) else e.item() for e in eigenvalues]
            
        return eigenvalues

    def compute_hessian(self, data, targets, top_n=5):
        """
        Compute Hessian eigenvalues and trace.
        
        Args:
            data: Input data tensor
            targets: Target tensor
            top_n: Number of top eigenvalues to return
            
        Returns:
            Dictionary with Hessian analysis results
        """
        hessian_comp = hessian(self.model, self.criterion, data=data, target=targets)
        
        # Get top eigenvalues and trace
        top_eigenvalues = hessian_comp.eigenvalues(top_n=top_n)
        trace = hessian_comp.trace()
        
        return {
            'top_eigenvalues': top_eigenvalues,
            'trace': trace
        }
    
    def plot_eigenvalue_distribution(self, eigenvalues, save_path):
        """Plot the distribution of eigenvalues."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(eigenvalues)), eigenvalues)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.title('Top Hessian Eigenvalues')
        plt.savefig(save_path)
        plt.close()
        
    def save_hessian_data(self, hessian_data, save_path):
        """Save Hessian analysis data to disk."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {
            'top_eigenvalues': hessian_data['top_eigenvalues'].tolist() if isinstance(
                hessian_data['top_eigenvalues'], np.ndarray) else hessian_data['top_eigenvalues'],
            'trace': float(hessian_data['trace']) if not isinstance(hessian_data['trace'], float) else hessian_data['trace']
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

class GradientTracker:
    """Tracks gradient statistics during training."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.gradients = {}
        self.hooks = []
        self.param_groups = {}
        self.gradient_stats = {
            'mean': [],
            'var': [],
            'norm': [],
            'layer_norms': {}
        }
            
    def get_gradient_stats(self):
        """Get gradient statistics."""
        stats = {}
        for name, param in self.model.named_parameters():
            print(name)
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                stats[name] = {
                    'mean': np.mean(grad_data),
                    'var': np.var(grad_data),
                    'norm': np.linalg.norm(grad_data)
                }   
                print(f"Gradient stats for {name}: mean={stats[name]['mean']}, var={stats[name]['var']}, norm={stats[name]['norm']}")
        # Compute overall statistics
        all_grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.model.parameters() if param.requires_grad and param.grad is not None])
        stats['overall'] = {
            'mean': np.mean(all_grads),
            'var': np.var(all_grads),
            'norm': np.linalg.norm(all_grads)
        }

        return stats
    
    def collect_gradient_stats(self):
        """Collect gradient statistics after backward pass."""
        all_grads = []
        layer_norms = {}
        
        # Collect all gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_data = param.grad.data.cpu().numpy().flatten()
                all_grads.append(grad_data)
                
                # Collect layer-specific gradients
                layer_name = name.split('.')[0]
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                
                layer_norms[layer_name].append(np.linalg.norm(grad_data))
        
        # Compute overall statistics
        all_grads = np.concatenate(all_grads)
        mean_grad = np.mean(all_grads)
        var_grad = np.var(all_grads)
        norm_grad = np.linalg.norm(all_grads)
        
        # Update statistics
        self.gradient_stats['mean'].append(mean_grad)
        self.gradient_stats['var'].append(var_grad)
        self.gradient_stats['norm'].append(norm_grad)
        
        # Update layer norms
        for layer_name, norms in layer_norms.items():
            if layer_name not in self.gradient_stats['layer_norms']:
                self.gradient_stats['layer_norms'][layer_name] = []
            
            self.gradient_stats['layer_norms'][layer_name].append(np.mean(norms))
        
        return {
            'mean': mean_grad,
            'var': var_grad,
            'norm': norm_grad,
            'layer_norms': {k: v[-1] for k, v in self.gradient_stats['layer_norms'].items()}
        }
    
    def save_gradient_stats(self, save_path):
        """Save gradient statistics to disk."""
        with open(save_path, 'w') as f:
            json.dump(self.gradient_stats, f, indent=4)

class AttentionVisualizer:
    """Visualizes attention patterns in transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.attention_weights = {}
        self.hooks = []  # Add hooks attribute
        
    def _register_hooks(self):
        """Register hooks to collect attention weights."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                hook = module.register_forward_hook(self._attention_hook)
                self.hooks.append(hook)  # Save hook reference
                
    def _attention_hook(self, module, input, output):
        """Hook function to store attention weights."""
        if hasattr(module, 'attn'):
            self.attention_weights[module] = module.attn.detach().cpu()

    def get_attn_map(self, input):
        output, attn = self.model(input, record_attn=True)
        return attn
            
    def plot_attention_heatmap(self, weights, save_path):
        """Plot attention weights as a heatmap."""
        if not weights:
            return False
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, cmap='viridis')
        plt.title('Attention Weights')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return True
    
    def collect_attention_weights(self):
        """Collect attention weights after forward pass."""
        return self.attention_weights
    
    def save_attention_data(self, save_path):
        """Save attention data to disk."""
        # Convert to serializable format
        serializable_data = {}
        for key, value in self.attention_weights.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
                
        with open(save_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)

class HiddenStateAnalyzer:
    """Analyzes hidden state representations."""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.hidden_states = {}
        self.hooks = []  # Add hooks attribute
        
    def _register_hooks(self):
        """Register hooks to collect hidden states."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: self._hidden_hook(m, i, o, name)
                )
                self.hooks.append(hook)  # Save hook reference
                
    def _hidden_hook(self, module, input, output, name):
        """Hook function to store hidden states."""
        self.hidden_states[name] = {
            'input': input[0].detach().cpu(),
            'output': output.detach().cpu()
        }
        
    def compare_with_solver(self, solver_outputs):
        """Compare model outputs with solver outputs."""
        comparisons = {}
        try:
            for state in self.hidden_states.values():
                if state['name'] == 'output_layer':
                    model_output = state['output'].numpy()
                    solver_outputs_np = np.array(solver_outputs)
                    
                    # Make sure shapes are compatible
                    if model_output.shape != solver_outputs_np.shape:
                        print(f"Shape mismatch: model_output {model_output.shape}, solver_outputs {solver_outputs_np.shape}")
                        
                        # Try to make them compatible
                        model_flat = model_output.flatten()[:min(model_output.size, solver_outputs_np.size)]
                        solver_flat = solver_outputs_np.flatten()[:min(model_output.size, solver_outputs_np.size)]
                        
                        cosine_sim = float(cosine_similarity([model_flat], [solver_flat])[0][0])
                        l2_dist = float(np.linalg.norm(model_flat - solver_flat))
                    else:
                        cosine_sim = float(
                            np.mean([cosine_similarity(m.reshape(1, -1), s.reshape(1, -1))[0][0] 
                                    for m, s in zip(model_output, solver_outputs_np)])
                        )
                        l2_dist = float(
                            np.mean([np.linalg.norm(m - s) for m, s in zip(model_output, solver_outputs_np)])
                        )
                    
                    comparisons['cosine_similarity'] = cosine_sim
                    comparisons['l2_distance'] = l2_dist
        except Exception as e:
            print(f"Error in compare_with_solver: {str(e)}")
            comparisons['cosine_similarity'] = 0.0
            comparisons['l2_distance'] = 0.0
            
        return comparisons 