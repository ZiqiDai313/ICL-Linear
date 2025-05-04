import torch
import numpy as np
import torch



def generate_pd_matrix_batch(batch_size, n, condition_number, alpha=1e-3, fix_min=True):
    # Generate a batch of random matrices B
    B = torch.randn(batch_size, n, n)
    # Step 1: Create a symmetric matrix A for each example in the batch
    A = torch.bmm(B, B.transpose(1, 2))
    # Step 2: Adjust to ensure positive definiteness

    A += alpha * torch.eye(n).unsqueeze(0)  # Broadcasting identity matrix
    
    # Step 3: Adjust for the desired condition number
    eigenvalues, Q = torch.linalg.eigh(A)
    
    if fix_min:
        min_eigenvalue = eigenvalues.min(dim=1, keepdim=True)[0]
        max_eigenvalue = min_eigenvalue * condition_number
    else:
        #print("take the maximum eigenvalues")
        max_eigenvalue = eigenvalues.max(dim=1, keepdim=True)[0]
        min_eigenvalue = max_eigenvalue / condition_number
    
    scaling = torch.linspace(0, 1, n).unsqueeze(0).expand(batch_size, -1)
    scaled_eigenvalues = min_eigenvalue + scaling * (max_eigenvalue - min_eigenvalue)
    
    A_scaled = torch.matmul(Q, scaled_eigenvalues.diag_embed())
    A_scaled = torch.matmul(A_scaled, Q.transpose(-1, -2))
    
    return A_scaled

def generate_linear_system_batch(batch_size, d, condition_number, mode='tf', alpha=1e-3, fix_min=True):
    A_batch = generate_pd_matrix_batch(batch_size, d, condition_number, alpha=alpha, fix_min=fix_min)
    
    
    x_batch = torch.FloatTensor(batch_size, d).normal_(0, 1)  #torch.randn(batch_size, d) #  
    b_batch = torch.einsum('bij,bj->bi', A_batch, x_batch)
    
    if mode == 'tf':
        X_test = torch.zeros(batch_size, 1, d)
        X = A_batch
        # A = B, N, d   X_comb = B, N+1, d
        X_comb= torch.cat([X, X_test],dim=1)
        # b_batch = B, N  b_comb = B, N+1
        b_test = torch.ones(batch_size, 1)
        b_comb = torch.cat([b_batch,b_test],dim=1)

        # Z = B, N+1, d+1
        Z = torch.cat([X_comb, b_comb.unsqueeze(-1)], dim=2)
        return Z, x_batch
    
    elif mode == 'solver':
        return A_batch, x_batch, b_batch
    

def convert_data_solver_to_tf(A_batch, x_batch, b_batch):
    batch_size = A_batch.shape[0]
    d = A_batch.shape[-1]
    X_test = torch.zeros(batch_size, 1, d)
    # A = B, N, d
    X = A_batch
    #X_comb = B, N+1, d
    X_comb= torch.cat([X,X_test],dim=1)
    # b_batch = B, N  b_comb = B, N+1
    b_test = torch.ones(batch_size, 1)
    b_comb = torch.cat([b_batch,b_test],dim=1)

    # Z = B, N+1, d+1
    Z = torch.cat([X_comb, b_comb.unsqueeze(-1)], dim=2)
    return Z, x_batch
    