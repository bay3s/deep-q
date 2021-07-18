import gym
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


class Environment:
  
  def __init__(self):
    self.env = gym.make('CartPole-v0').unwrapped
    self.env.reset()

    self.resize = T.Compose([T.ToPILImage(), T.Resize(40, interpolation = Image.CUBIC), T.ToTensor()])
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  def get_screen(self):
    screen = self.env.render(mode = 'rgb_array').transpose((2, 0, 1))

    # cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)

    scale = screen_width / self.env.x_threshold * 2
    cart_location = int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    if cart_location < view_width // 2:
      slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
      slice_range = slice(-view_width, None)
    else:
      slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    screen = np.ascontiguousarray(screen[:, :, slice_range], dtype = np.float32) / 255

    return self.resize(torch.from_numpy(screen)).unsqueeze(0).to(self.device)
