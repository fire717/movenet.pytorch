"""
@Fire
https://github.com/fire717
"""
import numpy as np
import cv2

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def generate_heatmap(x, y, size=(48,48), sigma=7):
    #heatmap, center, radius, k=1
    #centernet draw_umich_gaussian for not mse_loss
    k=1
    heatmap = np.zeros(size)
    height, width = size
    diameter = sigma#2 * radius + 1
    radius = sigma//2
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)


    tops = [[x,y],[30,30]]


    for top in tops:
        x,y = top

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    heatmap[heatmap>1] = 1
    return heatmap


def new_w():
    ratio = 0.01
    weights = np.zeros((48,48))
    half_w = []
    for i in range(24):
        half_w.append(1-i*ratio)
    line = np.array(half_w[::-1].copy()+half_w.copy())
    weights[23] = line.copy()
    weights[24] = line.copy()
    for i in range(1,24):
        weights[23-i] = line.copy()-i*ratio
        weights[24+i] = line.copy()-i*ratio

    return weights

new_w = new_w()
print(new_w[20:26,20:26])
print(new_w[:6,0:6])
cv2.imwrite("new_w.jpg", new_w*255)
np.save("my_weight_center.npy", new_w)


img = generate_heatmap(23,23)

print(img[20:26,20:26])
cv2.imwrite("t.jpg", img*255)

# w = np.load("center_weight_origin.npy").reshape(48,48)
# print(w[20:26,20:26])

res = img*new_w*10
print(res[20:26,20:26])

print(img.shape, new_w.shape, res.shape)