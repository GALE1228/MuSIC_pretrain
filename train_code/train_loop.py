import torch
import torch.optim as optim
import torch.nn as nn
from .metrics_utils import MLMetrics
import numpy as np
import argparse, os, copy
from tqdm import tqdm



def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

def mat2str(m):
    string=""
    if len(m.shape)==1:
        for j in range(m.shape[0]):
            string+= "%.3f," % m[j]
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                string+= "%.3f," % m[i,j]
    return string

def score(smooth_rate, output_sRBP, output_tRBP):
    prob = torch.sigmoid((smooth_rate * output_sRBP + (1 - smooth_rate) * output_tRBP))
    return prob

def train(model, device, train_loader,source_RBP_emb, target_RBP_emb,criterion, optimizer,batch_size, smooth_rate):

    model.train()
    met = MLMetrics(objective='binary')

    previous_loss_cls = 0
    previous_loss_smooth = 0

    for batch_idx, (x0, y0, y_s0) in enumerate(train_loader):
        x = x0.float().to(device)
        y = y0.float().to(device)
        y_s = y_s0.float().to(device)

        if y0.sum() == 0 or y0.sum() == batch_size:
            continue

        optimizer.zero_grad()

        output_sRBP = model(x, source_RBP_emb).squeeze(1)
        output_tRBP = model(x, target_RBP_emb).squeeze(1)

        loss_cls = criterion(output_sRBP, y)
        loss_smooth = criterion(output_tRBP, y_s)

        loss_cls.backward(retain_graph=True)
        grads_cls = [p.grad.clone() if p.grad is not None else None
                     for p in model.parameters()]

        loss_smooth.backward(retain_graph=True)
        grads_smooth = [p.grad.clone() if p.grad is not None else None
                        for p in model.parameters()]


        delta_smooth = loss_smooth.item() / previous_loss_smooth if previous_loss_smooth > 0 else 1
        delta_cls = loss_cls.item() / previous_loss_cls if previous_loss_cls > 0 else 1

        weight_cross_entropy = 1 / delta_cls
        weight_smooth = 1 / delta_smooth

        weight_cross_entropy_norm = weight_cross_entropy / (weight_cross_entropy + weight_smooth)
        weight_smooth_norm = weight_smooth / (weight_cross_entropy + weight_smooth)

        for p, g_cls, g_sm in zip(model.parameters(), grads_cls, grads_smooth):
            if g_cls is None:
                continue
            p.grad = weight_cross_entropy_norm * g_cls + weight_smooth_norm * g_sm

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        prob = torch.sigmoid(output_sRBP)
        y_np = y.detach().cpu().numpy().astype(int)
        p_np = prob.detach().cpu().numpy()

        loss_total = smooth_rate * loss_cls + (1 - smooth_rate) * loss_smooth
        met.update(y_np, p_np, [loss_total.item()])

        previous_loss_cls = loss_cls.item()
        previous_loss_smooth = loss_smooth.item()

    return met

def validate(model, device, test_loader, source_RBP_emb, target_RBP_emb, criterion ,smooth_rate):
    model.eval()
    y_all = []
    p_all = []
    l_all = []
    with torch.no_grad():
        for batch_idx, (x0, y0, y_s0) in enumerate(test_loader):
            x, y, y_s = x0.float().to(device), y0.to(device).float(), y_s0.to(device).float()
            
            output_sRBP = model(x, source_RBP_emb)
            output_sRBP = output_sRBP.squeeze(1)

            output_tRBP  = model(x, target_RBP_emb)
            output_tRBP = output_tRBP.squeeze(1)

            loss_cls = criterion(output_sRBP, y)
            loss_smooth = criterion(output_tRBP, y_s)
            
            prob = score(smooth_rate, output_sRBP, output_tRBP)
            
            loss_total = smooth_rate * loss_cls + (1 - smooth_rate) * loss_smooth
            y_np = y.to(device='cpu', dtype=torch.long).numpy()
            p_np = prob.to(device='cpu').numpy()
            l_np = loss_total.item()

            y_all.append(y_np)
            p_all.append(p_np)
            l_all.append(l_np)

    y_all = np.concatenate(y_all)
    p_all = np.concatenate(p_all)
    l_all = np.array(l_all)
    
    met = MLMetrics(objective='binary')
    met.update(y_all, p_all,[l_all.mean()])
     
    return met, y_all, p_all

def inference(args, model, device, test_loader, rna_names_all, smooth_rate, source_RBP_emb, target_RBP_emb):
    model.eval()
    p_all = []
    y_all = []
    rna_names_out = []
    with torch.no_grad():
        for batch_idx, (x0, y0) in enumerate(test_loader):
            x, y = x0.float().to(device), y0.to(device).float()

            output_sRBP = model(x, source_RBP_emb).squeeze(1)
            output_tRBP = model(x, target_RBP_emb).squeeze(1)

            prob = score(smooth_rate, output_sRBP, output_tRBP)

            p_np = prob.to(device='cpu').numpy()
            p_all.append(p_np)

            y_np = y.to(device='cpu').numpy()
            y_all.append(y_np)

            rna_names_out.extend(rna_names_all[batch_idx * test_loader.batch_size : (batch_idx + 1) * test_loader.batch_size])

    p_all = np.concatenate(p_all)
    y_all = np.concatenate(y_all)
    return p_all, y_all, rna_names_out

def compute_high_attention_region(args, model, device, test_loader, rna_names_all, target_RBP_emb):
    from model_code.smoothgrad import GuidedBackpropSmoothGrad
    model.eval()
    L = 20
    hars = []
    rna_names_out = []

    sgrad = GuidedBackpropSmoothGrad(model, device=device)
    for batch_idx, (x0, y0) in enumerate(test_loader):
        X, Y = x0.float().to(device), y0.to(device).float()
        output = model(X, target_RBP_emb)
        prob = torch.sigmoid(output)
        p_np = prob.detach().cpu().numpy().reshape(-1)
        rna_names_out_batch = rna_names_all[batch_idx * test_loader.batch_size : (batch_idx + 1) * test_loader.batch_size]
        rna_names_out.extend(rna_names_out_batch)

        guided_saliency  = sgrad.get_batch_gradients(X, target_RBP_emb, Y) # (N, 200, 1282)

        attention_region = guided_saliency.sum(dim=2).to(device='cpu').numpy()  # (N, 200)
        N, NS = attention_region.shape # (N, 200)

        for i in range(N):
            iar = attention_region[i]
            ar_score = np.array([iar[j:j+L].sum() for j in range(NS-L+1)])
            highest_ind = np.argmax(iar)
            
            rna_name = rna_names_out_batch[i]
            if isinstance(rna_name, bytes):
                rna_name = rna_name.decode('utf-8')

            hars.append("{}\t{:.6f}\t{}\t{}\n".format(rna_name, p_np[i], highest_ind, highest_ind+L))

    return hars
