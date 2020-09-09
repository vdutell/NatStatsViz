import numpy as np

def make_onef_amp(shape, amp_alpha=1, k=1):
    y, x = np.indices(shape)
    center = np.array(shape)/2
    r = np.hypot(x - center[1], y - center[0])
    amp = k/(1+r**amp_alpha)
    return amp

def make_onef_img_grey(shape, amp_alpha=1, k=1):
    onef_amp = make_onef_amp(shape, amp_alpha, k)
    rand_phase = np.random.rand(*shape)*2*np.pi - np.pi
    ampspec = onef_amp*np.exp(1j*rand_phase)
    img = np.real(np.fft.ifft2(np.fft.ifftshift(ampspec)))
    #remap to [0,1]
    img = img/np.max(img)
    return img

def make_onef_img_color(shape, amp_alpha=1, k=1):
    img = np.zeros((*shape,3))
    for i in range(3):
        img[:,:,i] = make_onef_img_grey(shape, amp_alpha, k)
    return img
        
