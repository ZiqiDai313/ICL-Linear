from data_solver import generate_linear_system_batch
import torch
import numpy as np
from baseline_solver import cgd
from scipy.sparse.linalg import cg
from collections import defaultdict
import sys
import os

test_seed=99
np.random.seed(test_seed)
torch.manual_seed(test_seed)

# Example usage
batch_size = 2000  # Number of examples in the batch
condition_number = 5
alpha=0.001
fix_min=False
cg_results = {}
test_data = {}
for dim in [5, 9, 10, 15, 20, 25]:
    print(dim)
    cg_results[f'dim_{dim}'] = {}
    A_batch, x_batch, b_batch = generate_linear_system_batch(batch_size=batch_size, 
                                                             d=dim, 
                                                             condition_number=condition_number, 
                                                             mode='solver', 
                                                             alpha=alpha, 
                                                             fix_min=fix_min)
    
    test_data[f'dim_{dim}'] = {'A_batch': A_batch, 'x_batch': x_batch, 'b_batch': b_batch}
    for iters in range(4, 11):
    # for i in range(4):
    #     print(torch.dot(A_batch[0, i, :], x_batch[0, :]), b_batch[0, i])
        test_loss = 0
        x_lst = []
        for i in range(len(A_batch)):
            x, rela_residuals = cgd(A_batch[i], b_batch[i], iters)
            
            x_lst.append(x.unsqueeze(0))
            test_loss += torch.nn.functional.mse_loss(x_batch[i], x).item()
        test_loss = test_loss / len(A_batch)
        
        x_lst = torch.cat(x_lst, dim=0)
        batch_mse_loss = ((x_lst - x_batch)**2).mean(dim=1).mean().item()
        
        print(test_loss, batch_mse_loss)
        cg_results[f'dim_{dim}'][f'iters_{iters}'] = (test_loss, batch_mse_loss)
        
os.makedirs(f'./data/nla/cg', exist_ok=True)
np.save(f'./data/nla/cg/test_alpha_{alpha}_fix_min_{fix_min}.npy', cg_results)

torch.save(test_data, f'./data/nla/test_data_alpha_{alpha}_fix_min_{fix_min}.pt')