import os, sys
import numpy as np
import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from scipy.misc import imresize
from PIL import Image
import numpy as np

def imresize(image, size):
    image_pil = Image.fromarray(image)
    return image_pil.resize(size, Image.Resampling.LANCZOS) 

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

def get_nt_height_for_str(pwm, height, norm):

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

        heights[:, i] = np.floor(np.sum(pwm[:, i]) * np.minimum(total_height, height * 2))

    heights_mean = np.mean(heights, axis=0, keepdims=True)
    heights_mean = np.where(heights_mean == 0, 1, heights_mean)
    return heights_mean.astype(int)

def seq_logo_for_str(pwm, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    heights = get_nt_height_for_str(pwm, height, norm)
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
            nt_img = imresize(chars[index[j]], (nt_width, nt_height[j]))
            nt_img = np.array(nt_img)
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

def seq_logo_raw(pwm, rna_base, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    num_seq = len(rna_base)

    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255

    base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    for i in range(num_seq):

        nt_base = rna_base[i]

        if nt_base == 'T':
            nt_base = 'U'

        nt_img = imresize(chars[base_dict[nt_base]], (nt_width, max_height))
        nt_img = np.array(nt_img) # (height,nt_width, 3)

        height_range = range(0, max_height)
        width_range = range(i * nt_width, i * nt_width + nt_width)

        for k in range(3):
            for m in range(len(width_range)):
                logo[height_range, width_range[m], k] = nt_img[:, m, k]

    return logo.astype(np.uint8)

def seq_logo(pwm, rna_base, height=30, nt_width=10, norm=0, alphabet='rna', colormap='standard'):

    heights = get_nt_height(pwm, height, norm)

    num_nt, num_seq = heights.shape # 1,200

    width = np.ceil(nt_width*num_seq).astype(int)
    
    max_height = height*2
    logo = np.ones((max_height, width, 3)).astype(int)*255

    base_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    for i in range(num_seq):
        nt_height = heights[0, i]
        nt_base = rna_base[i]

        if nt_base == 'T':
            nt_base = 'U'

        nt_img = imresize(chars[base_dict[nt_base]], (nt_width, nt_height))
        nt_img = np.array(nt_img) # (height,nt_width, 3)

        # print(f"nt_img shape at position {i}: {nt_img.shape}")

        height_range = range(max_height - nt_height, max_height)
        width_range = range(i * nt_width, i * nt_width + nt_width)

        # print(f"height_range: {height_range}")
        # print(f"width_range: {width_range}")

        for k in range(3):
            for m in range(len(width_range)):
                logo[height_range, width_range[m], k] = nt_img[:, m, k]

    return logo.astype(np.uint8), heights

def decode_str_2(m):
    seq = ""
    for i in range(m.shape[1]):
        if m[0, i] == 1:
            seq += 'U'
        elif m[1, i] == 1:
            seq += 'P'
    return seq

def plot_saliency(X, rna_seq, W, nt_width=300, norm_factor=1, str_null=None, outdir="results/"):

    plot_index = np.where(np.sum(X[:1280,:], axis=0)!=0)[0] 

    num_nt = len(plot_index) # 200

    # assert len(rna_seq) == num_nt, f"Error: Length of rna_seq ({len(rna_seq)}) does not match num_nt ({num_nt})"

    trace_width = num_nt*nt_width 
    trace_height = 400

    seq_str_mode = False

    if X.shape[0] > 1280:
        seq_str_mode = True

    img_seq_raw = seq_logo_raw(X[:1280, plot_index], rna_seq, height=nt_width, nt_width=nt_width)

    # Structure line
    if seq_str_mode:

        x_str = X[1280:, :]
        decoded_str = decode_str_2(x_str)
        str_raw_code = np.array([1 if ch == 'U' else 0 for ch in decoded_str]).reshape(1, -1)

        line_str_raw = np.zeros(trace_width)
        for v in range(str_raw_code.shape[1]):
            line_str_raw[v*nt_width:(v+1)*nt_width] = (1 - str_raw_code[0, v]) * trace_height
    
    # Sequence logo
    seq_sal = normalize_seq_pwm(W[:1280, plot_index], X[:1280, plot_index] , factor=norm_factor) # (1280,200)
    img_seq_sal_logo, heights = seq_logo(seq_sal, rna_seq, height=nt_width*5, nt_width=nt_width)
    # Sequence saliency logo
    heights_normalized = (heights - np.min(heights)) / (np.max(heights) - np.min(heights))
    heights_image = np.uint8(heights_normalized * 200)
    img_seq_sal = imresize(heights_image, size=(trace_width, trace_height)) # (h = 400, w = 200*100)

    if seq_str_mode:
        # Structure saliency logo
        str_sal_normalized = normalize_pwm(W[1280:, plot_index], factor=norm_factor)
        str_sal_mean = np.mean(str_sal_normalized, axis=0, keepdims=True)
        str_sal_normalized = np.uint8(str_sal_mean * 200)
        img_str_sal = imresize(str_sal_normalized, size=(trace_width, trace_height))
    
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