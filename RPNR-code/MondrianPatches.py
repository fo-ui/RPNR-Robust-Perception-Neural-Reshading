import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

class MondrianGenerator():
    def __init__(self, shape = (256, 256), n_patches = 10, min_patch_size = 50, shading_res = (2, 2), gloss_points = 2, gloss_max_radius = 50):
        self.shape = shape
        self.n_patches = n_patches
        self.min_patch_size = min_patch_size
        self.shading_res = shading_res
        self.gloss_points = gloss_points
        self.gloss_max_radius = gloss_max_radius
        self.transform = transforms.Compose([
            transforms.RandomRotation(180),
            transforms.CenterCrop(158),
            transforms.Resize(self.shape)
        ])
    
    def _mondrian(self):
        """
        Generate a Mondrian painting.
        """
        img = torch.ones((self.shape[0], self.shape[1], 3)) * torch.randint(0, 255, (3,)) / 255
        for _ in range(self.n_patches):
            # generate a random rectangle
            x = torch.randint(0, self.shape[0] - self.min_patch_size, (1,)).item()
            y = torch.randint(0, self.shape[1] - self.min_patch_size, (1,)).item()
            w = torch.randint(self.min_patch_size, self.shape[0] - x, (1,)).item()
            h = torch.randint(self.min_patch_size, self.shape[1] - y, (1,)).item()
            # generate a random color
            c = torch.randint(0, 255, (3,)) / 255
            # fill the rectangle with the color
            img[x:x+w, y:y+h] = c
        return img.permute(2, 0, 1)

    def _perlin(self):
        """https://gist.github.com/vadimkantorov/ac1b097753f217c5c11bc2ff396e0a57"""
        fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3
        delta = (self.shading_res[0] / self.shape[0], self.shading_res[1] / self.shape[1])
        d = (self.shape[0] // self.shading_res[0], self.shape[1] // self.shading_res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, self.shading_res[0], delta[0]), torch.arange(0, self.shading_res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(self.shading_res[0]+1, self.shading_res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:self.shape[0],:self.shape[1],0] + shift[0], grid[:self.shape[0],:self.shape[1], 1] + shift[1]  ), dim = -1) * grad[:self.shape[0], :self.shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:self.shape[0], :self.shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def _gloss(self):
        """
        Generate a gloss map.
        """
        img = torch.zeros((self.shape[0], self.shape[1], 1))
        # white(-ish) background
        c = torch.ones((1,)) * 0.7
        for _ in range(self.gloss_points):
            # generate a random point
            x = torch.randint(0, self.shape[0], (1,)).item()
            y = torch.randint(0, self.shape[1], (1,)).item()
            # draw a circle
            radius = torch.randint(0, self.gloss_max_radius, (1,)).item()
            for k in range(2 * radius + 1):
                for l in range(2 * radius + 1):
                    i = k - radius
                    j = l - radius
                    d = math.sqrt(i**2 + j**2)
                    if 0 < x+i < self.shape[0] and 0 < y + j < self.shape[1] and d < radius:
                        img[x+i, y+j] = torch.max( c * ( 1 - d / radius ), img[x+i, y+j])
        return img.permute(2, 0, 1)

    def __call__(self, batch_size = 1):
        X = torch.zeros((batch_size, 3, self.shape[0], self.shape[1]))
        Y = torch.zeros((batch_size, 4, self.shape[0], self.shape[1]))
        for i in range(batch_size):
            A = self.transform( self._mondrian() )
            S = self._perlin().unsqueeze(0) * 0.3 + 0.5
            # SG = self._perlin().unsqueeze(0) * 0.3 + 0.5
            # SB = self._perlin().unsqueeze(0) * 0.3 + 0.5
            # S = torch.cat((SR, SG, SB), dim = 0)
            X[i] = A * S
            Y[i] = torch.cat((A, S), dim = 0)
        return X, Y
import torch
import torch.nn as nn
import torch.nn.functional as F

# train the model
def train(model, device, optimizer, epochs = 20, batch_size = 32, batch_per_epoch = 100, track_fct = None):
    # create the generator
    MG = MondrianGenerator()
    # pbar
    pbar = tqdm(total = epochs * batch_per_epoch, position=0, leave=True)
    running_loss = 1
    loss_hist = []
    for epoch in range(epochs):
        for batch in range(batch_per_epoch):
            X, Y = MG(batch_size = batch_size)
            X = X.to(device)
            Y = Y.to(device)
            # train the model
            optimizer.zero_grad()
            Y_pred = model(X)
            # Extract A, S, G
            # A = Y_pred[:, :3, :, :]
            # S = Y_pred[:, 3:4, :, :]
            # G = Y_pred[:, 4:5, :, :]
            # reconstruct X_pred
            # X_pred = A * S + G
            # MSE
            loss = F.mse_loss(Y_pred, Y) # + F.mse_loss(X_pred, X)
            loss.backward()
            optimizer.step()
            running_loss = running_loss * 0.9 + loss.item() * 0.1
            pbar.set_description(f'Epoch : {epoch} Loss: {running_loss:.4f} Batch: {batch}/{batch_per_epoch}')
            loss_hist += [running_loss]
            pbar.update()
        if track_fct is not None:
            X, Y = MG(batch_size = 1)
            X = X.to(device)
            Y = Y.to(device)
            track_fct(model, X, Y)
    return loss_hist