""" Patterns suitable for showing on an LED string. Also, a simple tool for showing them animated in a notebook. No gamma correction or dimming is done here, as they should be custom to the light string used. To get the animations make sure you use the widget backend, with %matplotlib widget."""

import functools
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

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
    hsv = np.zeros((length, 3), dtype=float)
    hsv[:, 0] = np.linspace(0, 1, length, endpoint=False)
    hsv[:, 1] = saturation
    hsv[:, 2] = value
    return 256 * hsv_to_rgb(hsv)


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
    num_locations = locations.sum()
    hsv = np.ones((num_locations, 3), dtype=float)
    hsv[:, 0] = rng.random(num_locations)
    result = np.zeros((length, 3), dtype=float)
    result[locations] = hsv_to_rgb(hsv)
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
def waves(length=55, dt=0.05, renewal_rate=0.005, impulse=8, wave_speed=2, damping=0.3):
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
def bicubic_path(length=55, dt=0.05, time_between_targets=2, tension=0):
    """Move the bar through spacetime, interpolating so as to hit randomly chosen
    points at regular intervals while maintaining smoothness in space and time.

    The idea is to have a string-position parameter s with 0<s<3 and time parameter
    with 0<t<3. We specify the colors on the grid where s=0, 1, 2, 3 and t=0, 1, 2, 3.
    Then we interpolate values for s to fill each pixel for 0<s<3 and each time
    t=1, 1+dt, 1+2dt, ... 2. Once we reach t=2, we drop the values for t=0 and shift
    the grid down by 1, introducing new values for t=3.

    A better way to do this would be to interpolate the values at t=1 and 2, and use the
    values at t=0 and 3 to specify the derivatives at t=1 and 2. That would make the movement
    smooth in time as well as space -- currently it is only smooth in space and continuous
    in time. Since the derivative f_t(t=2, s) is cubic in s, it is completely specified by
    its values at four points, so it will be continuous when we shift the values.

    We can do that with a tension parameter, taking the slope at (say, s=0, t=1) to be
    df/dt(0, 1) = (1-tension)(f(0,2)-f(0,1))/2. When the tension is 0, this is exactly
    the slope of the secant line. When the tension is 1, the slope is just constrained
    to be zero at t=1 and t=2. When the tension is 0.5, the spline is a centripetal
    catmull-rom spline.

    We could do the same for s, which would allow us to have many control points
    along the path. For a string without many pixels, that would diminish the effect
    of the smoothness -- we would lose it to the discrete nature of the display.
    """
    rng = np.random.default_rng()
    hues = np.ones((4, 4, 3), dtype=float)
    hues[:, :, 0] = rng.random((4, 4))
    grid = hsv_to_rgb(hues)
    hues = np.ones((4, 3), dtype=float)

    powers = np.arange(4).reshape(-1, 1)
    X = np.arange(4) ** powers
    Xinv = np.linalg.inv(X)
    s = np.linspace(0, 3, length) ** powers
    Sinv_s = Xinv @ s
    t = np.linspace(1, 2, int(time_between_targets / dt), endpoint=False) ** powers
    Tinv_t = Xinv @ t
    while True:
        paths = 256 * np.einsum("ij, jkl, km -> iml", Tinv_t.T, grid, Sinv_s)
        for path in paths:
            yield path
            time.sleep(dt)
        grid = np.roll(grid, shift=-1, axis=0)
        hues[:, 0] = rng.random(4)
        grid[-1] = hsv_to_rgb(hues)
