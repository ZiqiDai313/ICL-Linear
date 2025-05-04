from data_solver import generate_linear_system_batch, convert_data_solver_to_tf
import torch
import sys
import numpy as np
from baseline_solver import cgd
from scipy.sparse.linalg import cg
from collections import defaultdict
import argparse
import os
import time
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from linear_transformer import in_context_loss, Transformer_F, Transformer_F_w_embed, Transformer_F_w_embed_mlp, Transformer_F_w_embed_mlp_v2, Transformer_F_w_embed_mlp_double
from pyhessian import hessian
from analysis_utils import HessianAnalyzer, GradientTracker, AttentionVisualizer, HiddenStateAnalyzer
from optim import Shampoo, Muon, SOAP

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--log_dir', type=str, default='./checkpoints/nla/solver/' , help='')
parser.add_argument('--only_eval', type=str2bool, default=False)
parser.add_argument('--model', type=str, default='linear')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alg', type=str, default='adam')
parser.add_argument('--dim', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=2000)
parser.add_argument('--max_iters', type=int, default=10000)
parser.add_argument('--condition_number', type=int, default=5)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--clip', type=float, default=10)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--clip_type', type=str, default='all')
parser.add_argument('--hidden_dim_factor', type=int, default=3)
parser.add_argument('--scheduler', type=str, default='step')
parser.add_argument('--var', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--fix_min', type=str2bool, default=True)

parser.add_argument('--eval_hessian', type=str, default='False', help='Evaluate hessian')
parser.add_argument('--hessian_interval', type=int, default=2000, help='Evaluate hessian every x epochs')
parser.add_argument('--mini_hessian_batch_size',
                    type=int,
                    default=200,
                    help='input batch size for mini-hessian batch (default: 200)')
parser.add_argument('--hessian_batch_size',
                    type=int,
                    default=200,
                    help='input batch size for hessian (default: 200)')

# soap params
parser.add_argument('--precondition_frequency', type=int, default=50, help='Precondition frequency for SOAP')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cur_dir = args.log_dir #os.path.join(log_dir, exp_dir)

N = args.dim     # context length
var = args.var  # initializations scale of transformer parameter
shape_k = 0.1  # shape_k: parameter for Gamma distributed covariates
hist_stride = 1  # stride for saved model paramters in `train.ipynb'
stride = 100
# a convenience function for taking a step and clipping
def clip_wo_step(allparam, optimizer, clip_r = None):
    norm_p=None
    grad_all = allparam.grad
    if clip_r is not None:
        norm_p = grad_all.norm().item()
        if norm_p > clip_r:
            grad_all.mul_(clip_r/norm_p)

    return norm_p

# a convenience function for taking a step and clipping
def clip_wo_step_nested(allparam, optimizer, clip_r = None):
    norm_p=None
    grad_all = allparam.grad
    if clip_r is not None:
        for l in range(grad_all.shape[0]):
            for h in range(grad_all.shape[1]):
                for t in range(grad_all.shape[2]):
                    norm_p = grad_all[l,h,t,:,:].norm().item()
                    if norm_p > clip_r:
                        grad_all[l,h,t,:,:].mul_(clip_r/norm_p)

    return norm_p

cur_dir = cur_dir + \
    f'double_alpha_{args.alpha}_fix_min{args.fix_min}_seed_{args.seed}_{args.model}_var{args.var}_lr{args.lr}_{args.scheduler}_clip_{args.clip_type}_{args.clip}_head{args.n_head}_hidden_f_{args.hidden_dim_factor}_{args.alg}_dim{args.dim}_bs{args.batch_size}_iter{args.max_iters}_condnum{args.condition_number}_ly{args.n_layer}_context_{N}'
hist_dict = {'train_loss': [], 'iter_list':[], 'norms':[]}
os.makedirs(cur_dir, exist_ok=True)

f = open(cur_dir + '/train.log', "w", 1)
sys.stdout = f

criterion = nn.MSELoss()

test_data_dict = torch.load(f'./data/nla/test_data_alpha_{args.alpha}_fix_min_{args.fix_min}.pt')
test_data = test_data_dict['dim_9'] #
A_batch_test, x_batch_test, b_batch_test = test_data['A_batch'], test_data['x_batch'], test_data['b_batch']
Z_test, x_batch_test = convert_data_solver_to_tf(A_batch_test, x_batch_test, b_batch_test)
Z_test = Z_test.double().to(device)
x_batch_test = x_batch_test.double().to(device)


if not args.only_eval:
    seeds = [args.seed] #for demonstration purpose, just use 3 seeds
    keys = [(s,) for s in seeds]
    for key in keys:
        sd = key[0]
        
        prob_seed = sd
        opt_seed = sd
        
        #set seed and initialize model
        print(f"Seed", opt_seed)
        torch.manual_seed(opt_seed)
        
        if args.model == 'linear_w_embed':
            model_arch = Transformer_F_w_embed
            print("var", var, "hidden_dim", args.dim*args.hidden_dim_factor)
            model = model_arch(args.n_layer, args.n_head, args.dim, var, hidden_dim=args.dim*args.hidden_dim_factor, device=device)
        elif args.model == 'linear':
            model_arch = Transformer_F
            model = model_arch(args.n_layer, args.n_head, args.dim, var)
        elif args.model == 'linear_w_embed_mlp':
            model_arch = Transformer_F_w_embed_mlp_double
            model = model_arch(args.n_layer, 
                               args.n_head, 
                               args.dim, 
                               var, 
                               hidden_dim=args.dim*args.hidden_dim_factor, 
                               device=device)
        elif args.model == 'linear_w_embed_mlp_v2':
            model_arch = Transformer_F_w_embed_mlp_v2
            model = model_arch(args.n_layer, 
                               args.n_head, 
                               args.dim, 
                               var, 
                               hidden_dim=args.dim*args.hidden_dim_factor, 
                               device=device)
        
        
        model = model.double()
        model.to(device)

        if args.eval_hessian == 'True':
            Z, x_batch = generate_linear_system_batch(args.batch_size, 
                                                      args.dim, 
                                                      args.condition_number, 
                                                      mode='tf',
                                                      alpha=args.alpha, 
                                                     fix_min=args.fix_min)
            Z = Z.double()
            x_batch = x_batch.double()
            subset_size = min(500, Z.shape[0])  # Smaller subset size
            Z_subset = Z[:subset_size]
            x_batch_subset = x_batch[:subset_size]
            hessian_dataloader = (Z_subset.to(device), x_batch_subset.to(device))
            
            # hessian_analyzer = HessianAnalyzer(model, criterion)
            gradient_tracker = GradientTracker(model)
            attention_visualizer = AttentionVisualizer(model)
            hidden_state_analyzer = HiddenStateAnalyzer(model)
       
        if args.alg == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
        elif args.alg == 'adam':
            print(args.alg, 'alg adam')
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0)
        elif args.alg == 'adagrad':
            print(args.alg, 'alg adagrad')
            optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=0)
        elif args.alg == 'soap':
            optimizer = SOAP(model.parameters(), lr=args.lr, betas=(.95, .95), weight_decay=0, precondition_frequency=args.precondition_frequency)
        else: 
            assert False
            
            
        if args.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iters)
        
        np.random.seed(prob_seed)
        torch.manual_seed(prob_seed)
        
        for t in range(args.max_iters):
            model.train()
            start = time.time()
            Z, x_batch = generate_linear_system_batch(args.batch_size, 
                                                      args.dim, 
                                                      args.condition_number, 
                                                      mode='tf',
                                                      alpha=args.alpha, 
                                                     fix_min=args.fix_min)
            Z = Z.double()
            x_batch = x_batch.double()
            Z = Z.to(device)
            x_batch = x_batch.to(device)
            
            if args.scheduler == 'step':
                if t%5000==0 and t>1:# and t < 200001:
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
            elif args.scheduler == 'cosine':
                scheduler.step()
                
                
            N = Z.shape[1] - 1
            output = model(Z)
            
            loss = criterion(output[:, N, :args.dim], x_batch)
            #print("---------------------->",output.shape, N, args.dim, output[:, N, :args.dim].shape, x_batch.shape)
            loss.backward()

            if args.clip_type == 'all-att':
                print('all-att')
                norms = clip_wo_step(model.allparam, optimizer, clip_r=args.clip)
            elif args.clip_type == 'nested-att':
                print('nested-att')
                norms = clip_wo_step_nested(model.allparam, optimizer, clip_r=args.clip)
            elif args.clip_type == 'all':
                #print("clip_all")
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.clone().view(-1))
                gradient = torch.cat(grads)
                norms = torch.norm(gradient).item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
            else:
                if t == 0:
                    print("skip the clip")
                grads = []
                for param in model.parameters():
                    grads.append(param.grad.clone().view(-1))
                gradient = torch.cat(grads)
                norms = torch.norm(gradient).item()
                pass
                
            optimizer.step()
            optimizer.zero_grad()
            
            end=time.time()
            
            hist_dict['train_loss'].append(loss.item())
            hist_dict['iter_list'].append(t)
            hist_dict['norms'].append(norms)
            if t%100 ==0 or t<5:
                model.eval()
                output_test = model(Z_test)
                N = Z_test.shape[1] - 1
                test_loss = criterion(output_test[:, N, :args.dim], x_batch_test)
                print('iter {} | Train Loss: {}  Test Loss: {} time: {}  gradnorm: {}  lr: {}'.format(t, loss.item(), test_loss.item(), end-start, norms, optimizer.param_groups[0]['lr']))
            
            if t%1000 ==0 or t == args.max_iters-1:
                filename = f'/linearTF_ebed_iter_{t}.pth'
                filename = (cur_dir + filename)
                #print(filename)
                torch.save({'state_dict':model.state_dict()}, filename)
                np.save(f'{cur_dir}/train_hist_dict.npy', hist_dict)

            if args.eval_hessian == 'True' and t % args.hessian_interval == 0:
                subset_size = min(16, Z.shape[0])  # Smaller subset size
                Z_subset = Z[:subset_size]
                x_batch_subset = x_batch[:subset_size]

                print('Computing hessian for epoch 0')
                # evaluate hessian
                assert hessian_dataloader is not None, 'No hessian data found'
                model.eval()
                criterion = nn.MSELoss()
                hessian_comp = hessian(model,
                                        criterion,
                                        data=hessian_dataloader,
                                        cuda=True)

                top_eigenvalues, _ = hessian_comp.eigenvalues()
                trace = hessian_comp.trace()
                density_eigen, density_weight = hessian_comp.density()

                hessian_stats = {
                    'top_eigenvalues': top_eigenvalues,
                    'trace': trace,
                    'density_eigen': density_eigen,
                    'density_weight': density_weight
                }
                
                # Process eigenvalues to floats
                def process_eigenvalues(e):
                    if isinstance(e, (list, tuple)):
                        result = []
                        for v in e:
                            result.extend(process_eigenvalues(v))
                        return result
                    elif isinstance(e, torch.Tensor):
                        return [float(e.item())]
                    elif isinstance(e, (int, float)):
                        return [float(e)]
                    else:
                        return [0.0]  # Default for invalid values
                
                os.makedirs(os.path.join(cur_dir, 'analysis/hessian'), exist_ok=True)
                os.makedirs(os.path.join(cur_dir, 'analysis/gradients'), exist_ok=True)
                os.makedirs(os.path.join(cur_dir, 'analysis/attention'), exist_ok=True)
                os.makedirs(os.path.join(cur_dir, 'analysis/hidden_states'), exist_ok=True)

                print(f"Top eigenvalues: {top_eigenvalues}")
                np.save(os.path.join(cur_dir, f'analysis/hessian/hessian_stats_epoch_{t}.npy'), hessian_stats)
                print(f"Saved Hessian data for iteration {t}")
                
                # Attention visualization
                print(f"Generating attention maps at iteration {t}...")
                attn_map = attention_visualizer.get_attn_map(
                    Z_subset
                )
                np.save(os.path.join(cur_dir, f'analysis/attention/attn_map_{t}.npy'), attn_map)
                print(f"Saved attention maps for iteration {t}")

            # flush the output buffer
            f.flush()



if args.model == 'linear_w_embed':
    model_arch = Transformer_F_w_embed
    print('linear_w_embed')
    model = model_arch(args.n_layer, args.n_head, args.dim, var, hidden_dim=args.dim*args.hidden_dim_factor, device=device)
elif args.model == 'linear':
    model_arch = Transformer_F
    model = model_arch(args.n_layer, args.n_head, args.dim, var)
elif args.model == 'linear_w_embed_mlp':
    model_arch = Transformer_F_w_embed_mlp_double
    model = model_arch(args.n_layer, args.n_head, args.dim, var, hidden_dim=args.dim*args.hidden_dim_factor, device=device)

model = model.double()
model.to(device)
N = args.dim

test_data_dict = torch.load(f'./data/nla/test_data_alpha_{args.alpha}_fix_min_{args.fix_min}.pt')
test_data = test_data_dict['dim_9'] #
A_batch_test, x_batch_test, b_batch_test = test_data['A_batch'], test_data['x_batch'], test_data['b_batch']
Z_test, x_batch_test = convert_data_solver_to_tf(A_batch_test, x_batch_test, b_batch_test)
Z_test = Z_test.double().to(device)
x_batch_test = x_batch_test.double().to(device)
#list(range(0, 210, 1)) + list()

iter_lst = list(range(0, 101, 1)) + list(range(100, args.max_iters, 100))
mse_lst = []
torch_mse_lst = []
for t in iter_lst:
    print(t)
    state_dict = torch.load(f'{cur_dir}/linearTF_ebed_iter_{t}.pth')['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        output = model(Z_test)
        print(output.shape)
        x_lst = output[:, N, :args.dim]
        
        batch_mse_loss = ((x_lst - x_batch_test)**2).mean(dim=1).mean().item()
        
        torch_mse_loss = criterion(x_lst, x_batch_test).item()
        
        mse_lst.append(batch_mse_loss)
        torch_mse_lst.append(torch_mse_loss)
        
hist_dict = {'iter_lst': iter_lst, 'mse_loss_lst': mse_lst, 'torch_mse_loss_lst': torch_mse_lst}
    

np.save(f'{cur_dir}/test_hist_dict.npy', hist_dict)