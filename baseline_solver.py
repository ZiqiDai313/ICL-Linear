import torch


def cgd(A, b, nits):
    """
    reference: https://gregorygundersen.com/blog/2022/03/20/conjugate-gradient-descent/
    Run conjugate gradient descent to solve x in Ax = b.
    
    Args:
    - A (torch.Tensor): A symmetric positive definite matrix of shape (n, n).
    - b (torch.Tensor): The right-hand side vector of shape (n).
    - nits (int): Maximum number of iterations.
    - tol (float): Tolerance for stopping criterion based on relative residual.
    
    Returns:
    - x (torch.Tensor): Solution vector of shape (n).
    - rela_residuals (list of float): List of relative residuals for each iteration.
    """
    n = A.shape[0]
    x = torch.zeros(n, dtype=b.dtype, device=b.device)  # Initial guess x = 0
    r = b - torch.mv(A, x)  # Initial residual r = b - Ax
    d = r.clone()  # Initial direction d = r
    rr = torch.dot(r, r)  # Initial residual norm squared
    normb = torch.norm(b)  # Norm of b
    normb = normb if normb != 0 else torch.tensor(1.0, device=b.device)  # Avoid division by zero

    rela_residuals = [torch.norm(r).item() / normb.item()]  # Initial relative residual
    
    for i in range(nits):
        Ad = torch.mv(A, d)  # A * d
        alpha = rr / torch.dot(d, Ad)  # Scalar step size
        x = x + alpha * d  # Update x
        r = r - alpha * Ad  # Update residual
        rr_new = torch.dot(r, r)  # Updated residual norm squared
        
        # Check for convergence based on relative residuals
        rela_residuals.append(torch.norm(r).item() / normb.item())
        # if rela_residuals[-1] <= tol:
        #     break
        
        beta = rr_new / rr  # Scalar beta
        d = r + beta * d  # Update direction
        rr = rr_new  # Update rr

    return x, rela_residuals


def cgd_batch(A_batch, b_batch, nits, tol=1e-5):
    """
    Run conjugate gradient descent to solve x in Ax = b for a batch of linear systems.
    
    Args:
    - A_batch (torch.Tensor): A batch of symmetric positive definite matrices of shape (batch_size, n, n).
    - b_batch (torch.Tensor): A batch of right-hand side vectors of shape (batch_size, n).
    - nits (int): Maximum number of iterations.
    - tol (float): Tolerance for stopping criterion based on relative residual.
    
    Returns:
    - x_batch (torch.Tensor): Solution vectors of shape (batch_size, n).
    - rela_residuals (list of torch.Tensor): List of relative residuals for each iteration.
    """
    batch_size, n = b_batch.shape
    x_batch = torch.zeros_like(b_batch)  # Initial guess x = 0 for all systems
    r_batch = b_batch - torch.bmm(A_batch, x_batch.unsqueeze(-1)).squeeze(-1)  # Initial residuals
    d_batch = r_batch.clone()  # Initial directions
    rr_batch = torch.sum(r_batch * r_batch, dim=1)  # Initial residual norms squared
    normb_batch = torch.norm(b_batch, dim=1)  # Norms of b for each system
    normb_batch = torch.where(normb_batch == 0, torch.ones_like(normb_batch), normb_batch)  # Avoid division by zero

    rela_residuals = [torch.norm(r_batch, dim=1) / normb_batch]  # Initial relative residuals
    
    for i in range(nits):
        Ad_batch = torch.bmm(A_batch, d_batch.unsqueeze(-1)).squeeze(-1)  # A * d
        alpha_batch = rr_batch / torch.sum(d_batch * Ad_batch, dim=1)  # Scalar step size for each system
        x_batch = x_batch + alpha_batch.unsqueeze(-1) * d_batch  # Update x
        r_batch = r_batch - alpha_batch.unsqueeze(-1) * Ad_batch  # Update residuals
        rr_new_batch = torch.sum(r_batch * r_batch, dim=1)  # Updated residual norms squared
        
        # Check for convergence based on relative residuals
        rela_residuals.append(torch.norm(r_batch, dim=1) / normb_batch)
        if torch.all(rela_residuals[-1] <= tol):
            break
        
        beta_batch = rr_new_batch / rr_batch  # Scalar beta for each system
        d_batch = r_batch + beta_batch.unsqueeze(-1) * d_batch  # Update direction
        rr_batch = rr_new_batch  # Update rr

    return x_batch, rela_residuals