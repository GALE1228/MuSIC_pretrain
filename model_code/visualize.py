import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.misc import imresize

package_directory = os.path.dirname(os.path.abspath(__file__))
acgu_path = os.path.join(package_directory,'acgu.npz')
chars = np.load(acgu_path,allow_pickle=True)['data']

def normalize_pwm(pwm, factor=None, MAX=None):
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm*factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm

def normalize_seq_pwm(pwm_sal, pwm_raw ,factor=None, MAX=None):
    pwm = pwm_sal * pwm_raw + pwm_sal
    if MAX is None:
        MAX = np.max(np.abs(pwm))
    pwm = pwm/MAX
    if factor:
        pwm = np.exp(pwm * factor)
    norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
    return pwm/norm


def get_nt_height(pwm, height, norm):

    def entropy(p):
        s = 0
        for i in range(len(p)):
            if p[i] > 0:
                s -= p[i]*np.log2(p[i])
        return s

    num_nt, num_seq = pwm.shape
    heights = np.zeros((num_nt,num_seq))
    for i in range(num_seq):
        if norm == 1:
            total_height = height
        else:
            total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height
        
        heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2))

    return heights.astype(int)

def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    heights = get_nt_height(pwm, height, norm)
    num_nt, num_seq = pwm.shape
    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255
    for i in range(num_seq):
        nt_height = np.sort(heights[:,i])
        index = np.argsort(heights[:,i])
        remaining_height = np.sum(heights[:,i])
        offset = max_height-remaining_height

        for j in range(num_nt):
            if nt_height[j] <=0 :
                continue
            # resized dimensions of image
            nt_img = imresize(chars[index[j]], (nt_height[j], nt_width))
            # determine location of image
            height_range = range(remaining_height-nt_height[j], remaining_height)
            width_range = range(i*nt_width, i*nt_width+nt_width)
            # 'annoying' way to broadcast resized nucleotide image
            if height_range:
                for k in range(3):
                    for m in range(len(width_range)):
                        logo[height_range+offset, width_range[m],k] = nt_img[:,m,k]

            remaining_height -= nt_height[j]

    return logo.astype(np.uint8)


def decode_str_2(m):
    seq = ""
    for i in range(m.shape[1]):  # 遍历每列
        # 找到值为1的通道
        if m[0, i] == 1:
            seq += 'U'
        elif m[1, i] == 1:
            seq += 'P'
    return seq


def plot_saliency(X, W, nt_width=300, norm_factor=1, str_null=None, outdir="results/"):
    # Filter out zero-padding
    plot_index = np.where(np.sum(X[:4,:], axis=0)!=0)[0]
    # for i in range(10):
    #     idx = plot_index[i]
    #     base_idx = np.argmax(W[:4, idx])  # 找最大 saliency 的 base
    #     base = ['A', 'C', 'G', 'U'][base_idx]
    #     print(f"Position {idx}: Max saliency base = {base}, values = {W[:4, idx]}")
    num_nt = len(plot_index)
    trace_width = num_nt*nt_width
    trace_height = 400

    seq_str_mode = False

    if X.shape[0]>4:
        seq_str_mode = True

    # Sequence logo
    img_seq_raw = seq_logo(X[:4, plot_index], height=nt_width, nt_width=nt_width)
    
    if seq_str_mode:
        # Structure line
        x_str = X[4:, :]
        decoded_str = decode_str_2(x_str)
        str_raw_code = np.array([1 if ch == 'U' else 0 for ch in decoded_str]).reshape(1, -1)

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw_code.shape[1]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1 - str_raw_code[0, v]) * trace_height
    
    # Sequence saliency logo
    seq_sal = normalize_seq_pwm(W[:4, plot_index], X[:4, plot_index] , factor=norm_factor)
    img_seq_sal_logo = seq_logo(seq_sal, height=nt_width*5, nt_width=nt_width)
    img_seq_sal = imresize(W[:4, plot_index], size=(trace_height, trace_width))

    if seq_str_mode:
        # Structure saliency logo
        str_sal = W[4:, plot_index]  # 这里的 W[4, plot_index] 是 (2, 200) 矩阵
        str_sal_normalized = normalize_pwm(str_sal, factor=norm_factor)        
        img_str_sal_logo = seq_logo(str_sal_normalized, height=nt_width*5, nt_width=nt_width)        
        img_str_sal = imresize(str_sal_normalized, size=(trace_height, trace_width))
    
    # Plot
    fig = plt.figure(figsize=(10.1, 1.15))  # Increased the figure size for higher resolution
    gs = gridspec.GridSpec(nrows=4, ncols=1, height_ratios=[2.5, 1, 0.5, 1.2])  # Adjusted the ratio
    cmap_reversed = mpl.cm.get_cmap('jet')

    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.imshow(img_seq_sal_logo)
    plt.text(x=trace_width/2-50, y=10, s='MuSIC', fontsize=10, color='black')

    ax = fig.add_subplot(gs[1, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_sal, cmap=cmap_reversed)

    ax = fig.add_subplot(gs[2, 0]) 
    ax.axis('off')
    ax.imshow(img_seq_raw)

    if seq_str_mode:
        ax = fig.add_subplot(gs[3, 0]) 
        ax.axis('off')
        ax.imshow(img_str_sal, cmap=cmap_reversed)
        ax.plot(line_str_raw, '-', color='r', linewidth=1, scalex=False, scaley=False)
        
        # Plot black line to hide the -1(NULL structure score)
        x = (np.zeros(trace_width) + (1+0.01))*trace_height  +1.5
        ax.plot(x, '-', color='white', linewidth=1.2, scalex=False, scaley=False)
    
    # Use tight_layout() to minimize the margins
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)  # Reduce space around subplots

    # Save figure with high DPI for better resolution
    filepath = outdir
    fig.savefig(filepath, format='pdf', dpi=900 , bbox_inches='tight')  # Increase dpi to 600 for higher resolution
    plt.close('all')