""" Patterns suitable for showing on an LED string. Also, a simple tool for showing them animated in a notebook. No gamma correction or dimming is done here, as they should be custom to the light string used. To get the animations make sure you use the widget backend, with %matplotlib widget."""

import colorsys
import functools
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

patterns = {}


def pattern(f):
    """Register a pattern in `patterns`. Make sure the output has the right shape."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        for data in f(*args, **kwargs):
            yield np.clip(data, 0, 255).reshape(1, -1, 3).astype(int)

    patterns[f.__name__] = wrapper
    return wrapper


def show(pattern):
    fig, ax = plt.subplots()
    im = ax.imshow(next(pattern))

    # styling
    ax.set_axis_off()
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.toolbar_visible = False
    fig.set_figheight(1)
    fig.tight_layout()
    plt.show()

    # set the pattern to update in the background
    def update():
        for data in pattern:
            im.set_data(data)
            fig.canvas.draw_idle()

    thread = threading.Thread(target=update)
    thread.start()


def rainbow(length=55, saturation=0.75, value=1):
    data = np.zeros((length, 3), dtype=int)
    for i, hue in enumerate(np.linspace(0, 1, length)):
        data[i] = 256 * np.array(colorsys.hsv_to_rgb(hue, saturation, value))
    return data


@pattern
def cycle(data=rainbow(), shift=1, pause=0.25):
    while True:
        yield data
        data = np.roll(data, shift, axis=0)
        time.sleep(pause)


def laplace(data):
    """three-point stencil laplace operator with zero boundary condition"""
    lap = np.zeros_like(data)
    lap[1:-1] = data[2:] + data[:-2] - 2 * data[1:-1]
    lap[0] = data[1] - 2 * data[0]
    lap[-1] = data[-2] - 2 * data[-1]
    return lap


def color_impulse(length, rng, renewal_rate):
    """Make an impulse at each location, with probability 1-exp(-renewal_rate)
    so as to follow an exponential distribution. Choose the hue uniformly at random.
    """
    locations = rng.random(length) > np.exp(-renewal_rate)
    hue = rng.random(locations.sum())
    rgb = np.zeros((len(hue), 3), dtype=float)
    for i, h in enumerate(hue):
        rgb[i] = colorsys.hsv_to_rgb(h, 1, 1)
    result = np.zeros((length, 3), dtype=float)
    result[locations] = rgb
    return result


@pattern
def diffusion(
    length=55, dt=0.05, renewal_rate=0.1, impulse=2, diffusion_rate=5, damping=0.3
):
    """Models the heat equation with randomly placed impulse sources, and perfect sinks at the ends.
    Impulses arrive uniformly in space, following an exponential distribution in time.
    """
    data = np.zeros((length, 3), dtype=float)
    rng = np.random.default_rng()
    while True:
        data += impulse * color_impulse(length, rng, renewal_rate * dt)
        data += (diffusion_rate * laplace(data) - damping * data) * dt
        yield data * 256
        time.sleep(dt)


@pattern
def waves(length=55, dt=0.05, renewal_rate=0.01, impulse=8, wave_speed=2, damping=0.3):
    """Models the 1-dimensional wave equation."""

    rng = np.random.default_rng()
    position = np.zeros((length, 3), dtype=float)
    velocity = np.zeros((length, 3), dtype=float)
    while True:
        position += impulse * color_impulse(length, rng, renewal_rate * dt)
        velocity += dt * wave_speed**2 * laplace(position)
        position += (velocity - damping * position) * dt
        yield position**2 * 256
        time.sleep(dt)


@pattern
def bicubic_path(length=55, dt=0.05, time_between_targets=2, dimming=1):
    """Move the bar through spacetime, interpolating so as to hit randomly chosen
    points at regular intervals while maintaining smoothness in space and time.

    The idea is to have a string-position parameter s with 0<s<3 and time parameter
    with 0<t<3. To make the interpolant align on the grid points, enforce
    ends = T* A S, where the columns of S and T are powers of s and t. By solving
    for A on the grid points, we can compute the value elsewhere.

    We compute new lattice points whenever t reaches 2, shifting the old ones by 1.
    In this way, we always have 1<t<2 and we always have grid points balanced around
    our interval.
    """
    rng = np.random.default_rng()
    ends = rng.random((4, 4, 3)) * dimming
    powers = np.arange(4).reshape(-1, 1)
    X = np.arange(4) ** powers
    Xinv = np.linalg.inv(X)
    s = np.linspace(0, 3, length) ** powers
    Sinv_s = Xinv @ s
    t = np.linspace(1, 2, int(time_between_targets / dt), endpoint=False) ** powers
    Tinv_t = Xinv @ t
    while True:
        paths = 256 * np.einsum("ij, jkl, km -> iml", Tinv_t.T, ends, Sinv_s)
        for path in paths:
            yield path
            time.sleep(dt)
        ends = np.roll(ends, shift=-1, axis=0)
        ends[-1] = rng.random((4, 3)) * dimming
