import numpy as np
import matplotlib.pyplot as plt
import stftoolkit as stf
from scipy import optimize

def onef_func(x, alpha, k=1):
    epsilon=1
    if(alpha >=0):
        a = k/(x**alpha + epsilon)
    else:
        a = k * (x**(-1*alpha) + epsilon)
    return a

def onef_func_fitter(x, alpha):
    #helper function for fitting onef_func for k=1
    return(onef_func(x, alpha,k=1))

def make_onef_amp(shape, amp_alpha=1, k=1):
    y, x = np.indices(shape)
    center = np.array(shape)/2
    r = np.hypot(x - center[1], y - center[0])
    amp = onef_func(r, alpha=amp_alpha, k=k)
    return amp

def make_onef_img_grey(shape, amp_alpha=1, k=1):
    onef_amp = make_onef_amp(shape, amp_alpha, k)
    rand_phase = np.random.rand(*shape)*2*np.pi - np.pi
    ampspec = onef_amp*np.exp(1j*rand_phase)
    img = np.real(np.fft.ifft2(np.fft.ifftshift(ampspec)))
    #remap to [0,1]
    img = img - np.min(img)
    img = img/np.max(img)
    return img

def make_onef_img_color(shape, amp_alpha=1, k=1):
    img = np.zeros((*shape,3))
    for i in range(3):
        img[:,:,i] = make_onef_img_grey(shape, amp_alpha, k)
    return img

def calc_amp_img(img, alpha_guess=1, k_guess=None):
    amp2d = np.abs(np.fft.fftshift(np.fft.fft2(img)))
    amp1d = stf.azimuthalAverage(amp2d, 100, return_fqs=False)
    if(k_guess is not None):
        #this is somewhat of a hack due to discontinuity in function of positive and negative alpha values
        ampalpha_k_fit, std = optimize.curve_fit(onef_func, 1+np.arange(len(amp1d)), amp1d, p0=[alpha_guess, 1], maxfev=2000)
        ampalpha_fit, k_fit = ampalpha_k_fit   
    else:
        ampalpha_fit, std = optimize.curve_fit(onef_func_fitter, 1+np.arange(len(amp1d)), amp1d, p0=[alpha_guess], maxfev=2000)
        ampalpha_fit = ampalpha_fit[0]
        k_fit=1
    return(amp2d, amp1d, ampalpha_fit, k_fit)

###plotting functions

def plot_grey_color_ims(grey_img, color_img, amp_alpha):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(grey_img,cmap='Greys_r')
    plt.title(f'Greyscale, alpha={amp_alpha}')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(color_img)
    plt.title(f'Color, alpha={amp_alpha}')
    plt.axis('off')
    plt.savefig(f'output/AmpAlpha{amp_alpha}_Image.png')
    
def plot_grey_ampspec(twod_ampspec, oned_ampspec, goal_ampalpha, grey_ampalpha, grey_k, plot_true=True, fit=False):
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(np.log10(twod_ampspec))
    plt.title(f'2D Amp Spec (Log)')
    plt.subplot(1,2,2)
    plt.loglog(oned_ampspec,'.', label='measured_amp')
    if(plot_true):
        xvals = 1+np.arange(len(oned_ampspec))
        yvals = onef_func(xvals,alpha=goal_ampalpha, k=1)
        plt.loglog(xvals, yvals, label=f'goal: alpha={goal_ampalpha:0.2f}, k={1.:0.2f}')
    
    if(fit):
        xvals = 1+np.arange(len(oned_ampspec))
        yvals = onef_func(xvals,alpha=grey_ampalpha, k=np.max(twod_ampspec))
        plt.loglog(xvals, yvals, label=f'fit: alpha={grey_ampalpha:0.2f}, k={grey_k:0.2f}')
    plt.legend()
    plt.title(f'1D Amp Sepc')
    plt.savefig(f'output/AmpSpecAmpAlpha{goal_ampalpha}.png')
