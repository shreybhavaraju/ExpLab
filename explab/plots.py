# Shared plot style so the README figures look like they belong together.

import matplotlib.pyplot as plt

from explab.load import ROOT

FIG_DIR = ROOT / 'figures'

BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
INK, GRAY, LIGHT, BG = '#0b0b0b', '#52514e', '#d9d8d4', '#fcfcfb'


def setup():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.edgecolor': LIGHT,
        'axes.labelcolor': GRAY,
        'xtick.color': GRAY,
        'ytick.color': GRAY,
        'text.color': INK,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.color': '#ececea',
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
        'lines.linewidth': 2,
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.titlelocation': 'left',
        'legend.frameon': False,
    })


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
