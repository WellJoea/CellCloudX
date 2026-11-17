import numpy as np
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px #5.3.1
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scanpy as sc
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, to_hex


def colrows(ncell, nrows=None, ncols=None, soft=False):
    import math
    if (ncols is None) and (nrows is None):
        ncols = int(np.ceil(ncell**0.5))
        soft = True
    if not ncols is None:
        nrows = math.ceil(ncell/ncols)
        ncols = min(ncell, ncols)
    elif not nrows is None:
        ncols = math.ceil(ncell/nrows)
        nrows = min(ncell, nrows)
    if soft and ncell> 1 and (ncell - ncols*(nrows-1)<=1):
        ncols += 1
        nrows -= 1
    return (nrows, ncols)

def colorset(get='vega_20_scanpy'):
    #https://github.com/theislab/scanpy/commit/58fae77cc15893503c0c34ce0295dd6f67af2bd7
    #https://github.com/theislab/scanpy/issues/387
    from matplotlib import cm, colors
    '''
    Add = [ #0780cf - 765005 - fa6d1d - 0e2c82 - b6b51f - da1f18 - 701866 - f47a75 - 009db2 - 024b51 - 0780cf - 765005
            #6beffc
            #3b45dd
            #b240ce]
    '''
    vega_20 = list(map(colors.to_hex, cm.tab20.colors))
    #*vega_20[0:14:2], *vega_20[16::2],
    #*vega_20[1:15:2], *vega_20[17::2],
    COLORS = {
        'CellType_colors':[
            "#4bb4ff", '#ff7f0e', '#279e68', '#d62728', '#aa40fc', 
            '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', 
            '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
            '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31'],
        'tab20': list(map(colors.to_hex, cm.tab20.colors)),
        "vega_20" :[
            '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728',
            '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2',
            '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
            ],
        "vega_20_sc" : [ 
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31'],
        'CellTypeN_colors':[
            '#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', 
            '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', 
            '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
            '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
            '#6beffc', '#0e2c82', '#024b51'],

        'CellTypeT_colors':[
            '#1f77b4', '#0e2c82', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', 
            '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8', 
            '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
            '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
            '#6beffc', '#3b45dd', '#024b51'],
        
        'COL':['#9eebe2', '#f7bc13', '#ce9fb9', '#A52A2A', '#29b3d3', 
                    '#228B22', '#74404c', '#80959a', '#c8d59d' ],
        'COL1': ['#6ed0a7', '#2f3ea8', '#706c01', '#ad94ec'],
        'COL1':['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
                '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
                '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
                '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'],

        'COL2':['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8', 
                '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', 
                '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31'],

        'COL3':['#0780cf', '#765005', '#fa6d1d', '#0e2c82', '#b6b51f', 
                '#da1f18', '#701866', '#f47a75', '#009db2', '#024b51', 
                '#0780cf', '#765005', '#6beffc', '#3b45dd', '#b240ce'],
        
        'COL4':["#b240ce", "#b6b51f", "#0780cf", "#765005", "#fa6d1d", 
                "#0e2c82", "#da1f18", "#701866", "#f47a75", "#009db2", 
                "#024b51", "#0780cf", "#765005", "#6beffc", "#3b45dd", 
                '#ad94ec', '#00749d', '#6ed0a7', '#2f3ea8', '#706c01', 
                '#9be4ff', '#d70000'],

        'COL6':["#f47a75", "#fa6d1d", "#c93528", "#da1f18", '#d70000', 
                "#b240ce", "#701866", "#b6b51f", "#765005", "#009db2", 
                '#00749d', "#024b51", "#3b45dd", "#0e2c82"],

        'COL7':['#023fa5', '#7d87b9', '#bec1d4', '#d6bcc0', '#bb7784',
                '#8e063b', '#4a6fe3', '#8595e1', '#b5bbe3', '#e6afb9', 
                '#e07b91', '#d33f6a', '#11c638', '#8dd593', '#c6dec7', 
                '#ead3c6', '#f0b98d', '#ef9708', '#0fcfc0', '#9cded6',
                '#d5eae7', '#f3e1eb', '#f6c4e1', '#f79cd4', '#7f7f7f'],
        
        'sc_28': [ "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3", 
                            "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593", 
                            "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7", 
                            "#f3e1eb", "#f6c4e1", "#f79cd4", 
                            # these last ones were added: 
                            '#7f7f7f', "#c7c7c7", "#1CE6FF", "#336600",],

        # from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
        'sc_102': [ # "#000000",  # remove the black, as often, we have black colored annotation 
                    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
                      "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", 
                      "#997D87", "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601", "#3B5DFF", 
                      "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", 
                      "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", 
                      "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", 
                      "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", 
                      "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578",
                       "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C", 
                       "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", 
                       "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", 
                       "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", 
                       "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F", 
                       "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", 
                       "#0089A3", "#CB7E98", "#A4E804", "#324E72"],
        
        'my_cmap': continuous_cmap(["lightgrey", 'yellow', 'red','darkred']),
         "matcmap" : [  'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
            'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
            'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
            'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
            'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
            'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
            'tab20c',
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
            'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
            'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
            'turbo', 'nipy_spectral', 'gist_ncar'],
    } 
    if get is None:
        return COLORS
    else:
        ListedColormap(COLORS.get(get, vega_20))
        return(COLORS.get(get, vega_20))

def ListedColormap(cor_list, **kargs):
    return matplotlib.colors.ListedColormap(cor_list, **kargs)

def check_cmap( imaps):
    if isinstance(imaps, str):
        icmap = plt.get_cmap(imaps)
    elif isinstance(imaps, list):
        icmap = matplotlib.colors.ListedColormap(imaps)
    else:
        icmap = imaps
    return icmap


def adjust_colors(colors, dark=0.25, light=0.35):
    rgb = np.array([to_rgb(c) for c in colors])

    dark_colors = [
        to_hex(c * (1 - dark)) for c in rgb
    ]
    light_colors = [
        to_hex(c + (1 - c) * light) for c in rgb
    ]

    return dark_colors, light_colors


def cmap_legend(cmap, n, names=None, save=None, figsize=None, orantation='v'):
    icmap = check_cmap(cmap)
    if orantation == 'h':
        figsize = (min(n,10),1) if figsize is None else figsize
        value = np.arange(n).reshape(1,n)
    else:
        figsize = (1, min(n,10)) if figsize is None else figsize
        value = np.arange(n).reshape(n,1)

    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.pcolor(value, cmap=icmap)
    ax.set_aspect(1)

    if names is not None:
        if orantation == 'h':
            ax.get_yaxis().set_visible(False)
            ax.set_xticks(np.arange(n)+0.5)
            ax.set_xticklabels(names)
        else:
            ax.get_xaxis().set_visible(False)
            ax.set_yticks(np.arange(n)+0.5)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.yaxis.tick_right()
    
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
    else:
        return fig

def longcolors(ncolor):
    if ncolor  == 169:
        steps = [18, 10, 14, 14, 14, 14,12,12,11,14,12,12,11]
    else:
        assert ncolor>=13
        steps = np.round(np.linspace(0, ncolor, 14)).astype(np.uint64)
        steps = (steps[1:]-steps[:-1])
    collist = {  
        'Ast': cmaptolist('terrain', n_color=steps[0], end=0.8),
        'Endo': cmaptolist('Oranges', n_color=steps[1], start=0.2),
        'Enx1' : cmaptolist('winter_r', n_color=steps[2], start=0),
        'Enx2' : cmaptolist('Blues_r', n_color=steps[3], start=0.05, end=0.8),
        'Enx3' : cmaptolist('cool', n_color=steps[4], start=0),
        'Enx4' : cmaptolist('spring', n_color=steps[5], start=0.3),
        'Inn1' : cmaptolist('Reds_r', n_color=steps[6], start=0.2, end=0.8),
        'Inn2' : cmaptolist('RdPu', n_color=steps[7], start=0.2, end=0.8),
        'Micro' : cmaptolist('twilight_shifted', n_color=steps[8], start=0.2, end=0.8),
        'Opc'   : cmaptolist('cividis', n_color=steps[9], start=0.1, end=0.7),
        'Oligo' : cmaptolist('gnuplot2', n_color=steps[10], start=0.1, end=0.7),
        'VLMC1' : cmaptolist('gist_earth', n_color=steps[11], start=0.1, end=0.7),
        'VLMC2' : cmaptolist('summer_r', n_color=steps[12], start=0.1, end=0.9),
        'others': ['grey']}
    return sum(collist.values(),[])

def sccolor(adata, groupby):
    '''
    plt.figure(figsize=(8, 2))
    sc.pl._tools.scatterplots._get_palette(adata, 'SID').values()
    for i in range(28):
        plt.scatter(i, 1, c=sc.pl.palettes.default_20[i], s=200)
        plt.scatter(i, 1, c=sc.pl.palettes.default_102[i], s=200)
        sc.pl._tools.scatterplots._get_palette(adata, 'SID').values()
    plt.show()
    '''
    import scanpy as sc
    return sc.pl._tools.scatterplots._get_palette(adata, groupby)

def rgb_to_hex0(rgb):
    rgb = np.asarray(rgb)
    
    if rgb.ndim == 1:
        if np.max(rgb) <= 1:
            rgb = rgb * 255
        rgb = np.round(rgb).astype(np.uint8)
        assert np.max(rgb) <= 255
        if len(rgb) == 3:
            return '#%02X%02X%02X' % tuple(rgb)
        elif len(rgb) == 4:
            return '#%02X%02X%02X%02X' % tuple(rgb)
        else:
            raise ValueError('rgb should be of length 3 or 4')
    
    elif rgb.ndim == 2:
        assert rgb.shape[1] in (3, 4), 'rgb should have 3 or 4 columns'
        if np.max(rgb) <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)

        if rgb.shape[1] == 3:
            packed = (rgb[:, 0].astype(np.uint32) << 16) | \
                     (rgb[:, 1].astype(np.uint32) << 8)  | \
                      rgb[:, 2].astype(np.uint32)
            return [f'#{v:06X}' for v in packed]
        else: 
            packed = (rgb[:, 0].astype(np.uint32) << 24) | \
                     (rgb[:, 1].astype(np.uint32) << 16) | \
                     (rgb[:, 2].astype(np.uint32) << 8)  | \
                      rgb[:, 3].astype(np.uint32)
            return [f'#{v:08X}' for v in packed]
    
    else:
        raise ValueError('Input must be 1D or 2D array')

def rgb_to_hex(rgb):
    rgb = np.asarray(rgb)

    if rgb.ndim not in (1, 2):
        raise ValueError("rgb must be a 1D or 2D array")
    if rgb.shape[-1] not in (3, 4):
        raise ValueError("rgb must have 3 or 4 channels")

    if rgb.size and rgb.max() <= 1:
        rgb = rgb * 255

    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)

    if rgb.ndim == 1:
        return "#" + rgb.tobytes().hex().upper()

    digits = np.frombuffer(b"0123456789ABCDEF", dtype=np.uint8)
    n, c = rgb.shape

    result = np.empty((n, 1 + 2 * c), dtype=np.uint8)
    result[:, 0] = ord("#")
    result[:, 1::2] = digits[rgb >> 4]
    result[:, 2::2] = digits[rgb & 15]

    return (
        result.view(f"S{1 + 2 * c}")
        .ravel()
        .astype(f"U{1 + 2 * c}")
    )

def to_rgba(col):
    return mcolors.to_rgba(col)
def to_rgb(col):
    return mcolors.to_rgb(col)
def to_hex(col, **kargs):
    return mcolors.to_hex(col, **kargs)

def cmaptolist(cmap, n_color=None, spaces = None, torgb=False, start=0, end=1):
    if not n_color is None:
        colors = np.linspace(start, end, n_color, endpoint=True)
    elif not spaces is None:
        assert (max(spaces) <=1) and (min(spaces) >=0)
        colors = spaces
    else:
        raise ValueError('A least one of n_color and spaces is not None.')
    if type(cmap) is str:
        colors = matplotlib.colormaps.get(cmap)(colors) #get_cmap
    elif isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        colors = cmap(colors) 

    if torgb:
        return colors
    else:
        colors = np.round(colors[:,:3]*255).astype(np.int16)
        return list(map(lambda x: '#%02x%02x%02x'.upper() %tuple(x), colors))

def hex_revers(value):
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    rgb =  tuple(255- int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return ('#%02x%02x%02x' % rgb)
    
def hex_to_rgb(value, alpha=None):
    value = value.strip().lstrip("#")
    if len(value) not in (6, 8):
        raise ValueError("Color must be #RRGGBB or #RRGGBBAA")
    rgb = tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
    if alpha is None and len(value) == 8:
        alpha = int(value[6:8], 16)
    return (*rgb, alpha)
    return rgb if alpha is None else (*rgb, alpha)

def rgb_to_dec(value):
    return [v/256 for v in value]
        
def continuous_cmap_html(hex_list, float_list=None):
    ''' 
    creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map

    >>>hex_list = ['#0091ad', '#3fcdda', '#83f9f8', '#d6f6eb', '#fdf1d2', '#f8eaad', '#faaaae', '#ff57bb']
    >>>float_list = [0, 0.05, 0.5, 0.6, 0.85, 0.9, 0.92, 1]
    >>>cmap=continuous_cmap_html(hex_list,float_list=float_list )
    '''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def continuous_cmap_name(names='viridis', diversity=99, alpha=1, trans=np.log2, shift=1):
    import matplotlib
    import matplotlib.pyplot as plt 
    Nspace = list(np.linspace(0, 1,diversity))
    Lspace = list(trans(np.linspace(0,1,diversity) + shift))
    Carray = plt.get_cmap(names)(Lspace)
    return matplotlib.colors.ListedColormap(Carray)

def continuous_cmap(cor_list):
    '''
    >>>continuous_cmap(["lightgrey", "blue", "mediumblue",'red','yellow'])
    '''
    return matplotlib.colors.LinearSegmentedColormap.from_list("my_cmp", cor_list)

def alpha_curve(x, center=0.65, sharpness=18, alpha_min=0.0, alpha_max=1.0):
    x = np.asarray(x)
    x = np.clip(x, 0, 1)
    y = 1 / (1 + np.exp(-sharpness * (x - center)))
    y0 = 1 / (1 + np.exp(-sharpness * (0 - center)))
    y1 = 1 / (1 + np.exp(-sharpness * (1 - center)))
    y = (y - y0) / (y1 - y0)
    return alpha_min + (alpha_max - alpha_min) * y

def merge_cmap(cmaps, ncolors=255, n_melt=11, clips=None):
    if isinstance(ncolors, int):
        ncolors = [ncolors] * len(cmaps)
    assert len(ncolors) == len(cmaps)

    clist = []
    for i, (icmap, incol) in enumerate(zip(cmaps, ncolors)):
        if clips is None:
            start=0
            end=1
        else:
            start = clips[i][0]
            end = clips[i][1]
        clist.append(cmaptolist(icmap, incol,start=start, end=end, torgb=True))
    
    if n_melt >0:
        mlist = []
        for i in range(len(cmaps)-1):
            istart = clist[i][-1]
            iend = clist[i+1][0]
            alist= np.linspace(istart, iend, n_melt+2, endpoint=True)[1:-1]
            mlist.append(clist[i])
            mlist.append(alist)
            mlist.append(clist[i+1])
        clist = mlist
    colors = np.vstack(clist)
    return continuous_cmap(colors)

def matplotlib_to_plotly(cmap, ncolors=255, decimal=8, cmap_type='rgb'):
    h = np.linspace(0,1, ncolors)
    corlist = cmap(h)
    rgbs = np.uint8(corlist[:,:3] * 255)
    if cmap_type == 'rgb':
        cmapl = [ [ float(h[i]),  f'rgb{tuple(rgbs[i,:3].tolist())}'] for i in range(corlist.shape[0])]
    elif cmap_type == 'rgba':
        alpha = np.round(corlist[:,3], decimal)
        cmapl = [ [ float(h[i]),  f'rgba{tuple([*(rgbs[i,:3].tolist()), float(alpha[i])])}'] for i in range(corlist.shape[0])]
    return cmapl

def alphacmap(mycmap, alphas, name = None, N=None ):
    if name is None:
        my_name = mycmap.name + 'alpha'
    cmap = mycmap

    if (type(alphas) in [float]) or np.isscalar(alphas):
        assert 0 <= alphas <=1
        if N is None:
            N = cmap.N
        alphas = np.full(N, 1.0)
    else:
        N = len(alphas)
    
    my_cmap = cmap(np.arange(N))
    my_cmap[:,-1] = alphas
    my_cmap = ListedColormap(my_cmap, name= my_name)
    return my_cmap

def px_colormap():
    px_cor = {}
    for i in px.colors.named_colorscales():
        px_cor[i] = i
        px_cor[f'{i}_r'] = f'{i}_r'
    px_cor['cmap1']  = cmap1px
    px_cor['cmap1px'] = cmap1px
    px_cor['cmap2']  = cmap2px
    px_cor['cmap2px']= cmap2px
    return px_cor

def get_px_colormap(icolor):
    px_cor = px_colormap()
    if icolor in px_cor.keys():
        return px_cor.get(icolor)
    else:
        # return icolor
        # raise ValueError(f'No such color map {icolor} in px.colors.named_colorscales()')
        return matplotlib_to_plotly(icolor)

cmap0=continuous_cmap(["lightgrey", 'cyan', "mediumblue","blue",])
cmap1=continuous_cmap(["lightgrey", 'yellow', 'red','darkred'])
cmap2=continuous_cmap(["lightgrey", "blue", "mediumblue",'red','yellow'])
cmap3=continuous_cmap(["skyblue", "cyan", 'yellow', 'darkorange', 'red','darkred'])
cmap0a = alphacmap(cmap0, np.linspace(0, 1, 256))
cmap1a = alphacmap(cmap1, np.linspace(0, 1, 256))
cmap2a = alphacmap(cmap2, np.linspace(0, 1, 256))
cmap3a = alphacmap(cmap3, (np.linspace(0, 2.5, 256).clip(0,1.5))/1.5)
cmap4a = alphacmap(cmap3, (np.linspace(0, 1.25, 256).clip(0,1)))

cmap1px = matplotlib_to_plotly(cmap1)
cmap2px = matplotlib_to_plotly(cmap2)
cmap1pxa = matplotlib_to_plotly(cmap1a, cmap_type='rgba')
cmap2pxa = matplotlib_to_plotly(cmap2a, cmap_type='rgba')
cmap3pxa = matplotlib_to_plotly(cmap3a, cmap_type='rgba')
cmap4pxa = matplotlib_to_plotly(cmap4a, cmap_type='rgba')

cmapsym1 = continuous_cmap([ 'blue', "deepskyblue", 'cyan',"white", 'yellow', 'red', 'darkred']) 
cmapsym1a = alphacmap(cmapsym1, np.r_[np.linspace(1,0,128,endpoint=False), np.linspace(0,1,64), np.repeat(1,64)])
cmapsym1b = alphacmap(cmapsym1, np.r_[np.linspace(1,0,128,endpoint=False), np.linspace(0,1,128)])


def cmapsplit(colormap='viridis_r'):
    cmap=continuous_cmap(['white', 'mistyrose','purple','darkblue'])
    cmapsub = matplotlib.cm.get_cmap(colormap)([0.4, 0.7, 1])
    cmapsub = np.r_[ [[1,1,1,1], [1,1,0.6,1]], cmapsub]
    cmap=continuous_cmap(cmapsub)
    return cmap

def pxsetcolor(color, ctype='discrete', cmap=None):
    import plotly.express as px
    if color is None:
        return {}
    elif ctype=='discrete':
        if type(color)==str and color in dir(px.colors.qualitative):
            #px.colors.named_colorscales()
            COLOR = eval('px.colors.qualitative.%s'%color)
        elif type(color)==np.ndarray:
            COLOR = color.tolist()
        elif type(color)==list:
            COLOR = color
        elif type(color)==dict:
            COLOR = color.values()
        else:
            COLOR = color
        return {'color_discrete_sequence': COLOR}
    elif ctype=='continuous':
        if not cmap is None:
            icolor = cmap
        else:
            icolor = color

        if (type(icolor) == str) and (icolor in px.colors.named_colorscales()):
            icolor = icolor
        elif (type(icolor) == list):
            try:
                icolor = matplotlib_to_plotly(continuous_cmap(icolor))
            except:
                icolor = icolor
        else:
            try:
                icolor = matplotlib_to_plotly(icolor)
            except:
                icolor = icolor
        return {'color_continuous_scale': icolor}
    else:
        return {}

def extend_colormap(value, under='#EEEEEE', over=-0.2, bits=256, color_name='my_cmp'):
    import matplotlib
    med_col = value
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    under = 'white' if under is None else under
    rgb = [ int(value[i:i + lv // 3], 16)/bits for i in range(0, lv, lv // 3)]

    if (over is None) or ( type(over) is int and over==0):
        return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col]))
    elif type(over) is int:
        rgb = [ i + over/bits for i in rgb ]
    elif type(over) is float and (0<= abs(over) <= 1) :
        rgb = [ i + over for i in rgb ]
    rgb = [ np.median([0, i, 1]) for i in rgb ]
    return(matplotlib.colors.LinearSegmentedColormap.from_list(color_name, [under, med_col, rgb]))

def plot_color(cmap_dict, width=10, heigth_ratio=0.3, tight=True, show=True,
               save =None, aspect='auto', fontsize=9):

    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cmaps = {}


    nrows = len(cmap_dict)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * heigth_ratio
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(width, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, 
                        bottom=0.15 / figh,
                        left=0.2, right=0.99)

    for ax, name in zip(axs, cmap_dict.keys()):
        imaps =  cmap_dict[name]
        if isinstance(imaps, str):
            icmap = plt.get_cmap(imaps)
        elif isinstance(imaps, list):
            icmap = matplotlib.colors.ListedColormap(imaps)
        else:
            icmap = imaps
        if hasattr(icmap, 'N'):
            N = icmap.N
        else:
            N = 256

        gradient = np.linspace(0, 1, N)
        gradient = np.arange(N)
        gradient = np.vstack((gradient, gradient))

        ax.imshow(gradient, aspect=aspect, cmap=icmap)
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=fontsize,
                transform=ax.transAxes)

    for ax in axs:
        ax.set_axis_off()
    if tight:
        fig.tight_layout()
    if save:
        if tight:
            fig.savefig(save, bbox_inches='tight')
        else:
            fig.savefig(save)
    if show is None:
        return fig, axs
    elif show is True:
        plt.show()
    else:
        plt.close()
    # cmaps.update(cmap_dict) 
    # return cmaps

def adata_color(adata, value=None, values=None,  inplace=False, palette=None, cmap =None, return_color=False):
    cmap = cmap or 'viridis'

    if not value is None:
        adata.obs[value] = adata.obs[value].astype('category')
        n_group = adata.obs[value].unique().shape[0]
        # n_group = np.unique(adata.obs[value]).shape[0] # NaN error
        if (inplace) or \
            (not f'{values}_colors' in adata.uns.keys()) or \
            (len(adata.uns[f'{values}_colors']) <n_group):
            if n_group >= 100:
                # colors =cmaptolist(cmap, n_group)
                colors = longcolors(n_group)
                adata.uns[f'{value}_colors'] = colors
            else:
                sc.pl._tools.scatterplots._get_palette(adata, value, palette=palette)
    else:
        if not values is None:
            value = "pie" if value is None else value
            celllen = adata.shape[0]
            add_col = int(np.ceil(celllen/len(values)))
            add_col = values * add_col 
            adata.obs[value] = pd.Categorical(add_col[:celllen], categories=values)
        sc.pl._tools.scatterplots._get_palette(adata, value, palette=palette)
    if return_color:
        return adata.uns[f'{value}_colors']


def get_contrasting_text_color(bg_hex):
    bg_hex = bg_hex.lstrip('#')
    r, g, b = int(bg_hex[0:2], 16), int(bg_hex[2:4], 16), int(bg_hex[4:6], 16)
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#FFFFFF" if brightness < 128 else "#000000"

def generate_color(index=1, total=1):
    import colorsys
    hue = (index / total) if total > 0 else 0
    r, g, b = colorsys.hls_to_rgb(hue, 0.75, 0.6)
    return '#{:02X}{:02X}{:02X}'.format(int(r*255), int(g*255), int(b*255))

def inverse_colors(colors):
    return ['#%02x%02x%02x' % (255 - int(color[1:3], 16), 
                                255 - int(color[3:5], 16), 
                                255 - int(color[5:7], 16))
             for color in colors]

def random_colors0(n, seed=None):
    rng = np.random.default_rng(seed=seed)
    colors = []
    while len(colors) < n:
        colors += np.random.randint(0, 0xFFFFFF, n-len(colors)).tolist()
        colors = list(set(colors))
    colors = ['#%06X' % color for color in colors]
    return colors


import numpy as np


def random_colors(
    n, min_dist=0.15, tohex=True, rgba=False,
    batch_size=50, max_attempts=100,
    seed=None, vmin=0, vmax=255,
):
    rng = np.random.default_rng(seed)
    k = n + batch_size * max_attempts

    colors = (
        np.column_stack([rng.permutation(k) for _ in range(3)])
        + rng.random((k, 3))
    ) / k

    selected = np.empty((n, 3))
    dmin = np.full(k, np.inf)
    idx = rng.integers(k)
    actual_dist = np.inf

    for i in range(n):
        if i:
            actual_dist = min(actual_dist, dmin[idx])

        selected[i] = colors[idx]
        dist = np.sum((colors - selected[i]) ** 2, axis=1)
        dmin = np.minimum(dmin, dist)
        dmin[idx] = -1
        idx = np.argmax(dmin)

    actual_dist = np.sqrt(actual_dist)
    if n > 1 and actual_dist < min_dist:
        print(
            f"minimum distance is {actual_dist:.3f}; "
            "increase batch_size or max_attempts."
        )

    if rgba:
        selected = np.c_[selected, np.ones(n)]

    if not tohex:
        return selected

    selected = np.clip(
        selected * 255, vmin, vmax
    ).round().astype(np.uint8)

    return [
        "#" + "".join(f"{v:02X}" for v in color)
        for color in selected
    ]


def color_palette(len_color, use_random=False, seed = None):
    #plt.colormaps()
    #plt.get_cmap('tab20')
    if use_random:
         return random_colors(len_color, seed=seed)
    if len_color <= 20:
        palette = sc.pl.palettes.default_20
    elif len_color <= 28:
        palette = sc.pl.palettes.default_28
    elif len_color <= len(sc.pl.palettes.default_102):  # 103 colors
        palette = sc.pl.palettes.default_102
    elif len_color <= 150:  # 103 colors
        palette = longcolors(len_color)
    else:
        palette = random_colors(len_color, seed=seed)

    return palette[:len_color]


def show_colors(
    colors,
    nrow=None,
    ncol=None,
    marker="o",
    size=1600,
    gap=1.0,
    edgecolor="black",
    linewidth=1.0,
    linestyle="-",
    names=None,
    names_size=12,
    names_pad=6,
    figsize=None,
    save=None,
    dpi=1500,
    transparent=True,
    show=True,
    pl=None,
    names_pos="bottom",
):
    """Draw colors from left to right, then top to bottom.

    nrow, ncol: Grid dimensions; missing values are inferred.
    marker: Any Matplotlib scatter marker, e.g. 'o', 's', '^', 'D', 'h'.
    size: Scatter marker size in points squared.
    gap: Center spacing in data units; scalar or (horizontal, vertical).
    edgecolor, linewidth, linestyle: Marker border settings.
    names: One label per color, or None to hide labels.
    names_pos: Label position: 'top', 'bottom', 'left', 'right', or 'center'.
    names_size: Label font size in points.
    names_pad: Padding from the marker border in points; ignored for 'center'.
    figsize: Figure size in inches; used only when pl is None.
    save: Output path or None; saves the entire figure.
    dpi, transparent: Output resolution and background transparency.
    show: Whether to call plt.show(). Use False when composing subplots.
    pl: Existing Matplotlib Axes or None to create a new figure.
    Returns: (fig, ax).
    """
    colors = list(colors)
    n = len(colors)
    if not n:
        raise ValueError("colors must not be empty.")
    if names is not None and len(names) != n:
        raise ValueError("names and colors must have the same length.")
    if nrow is None and ncol is None:
        ncol = int(np.ceil(np.sqrt(n)))
    if nrow is None:
        nrow = int(np.ceil(n / ncol))
    if ncol is None:
        ncol = int(np.ceil(n / nrow))
    if nrow * ncol < n:
        raise ValueError("nrow * ncol must be at least len(colors).")

    dx, dy = np.broadcast_to(gap, (2,))
    row, col = np.divmod(np.arange(n), ncol)
    x, y = col * dx, (nrow - 1 - row) * dy
    width = (ncol - 1) * dx + 1
    height = (nrow - 1) * dy + 1

    if pl is None:
        if figsize is None:
            figsize = (max(2, width * 1.1), max(1.5, height * 1.1 + 0.5))
        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    else:
        ax = pl
        fig = ax.figure

    points = ax.scatter(
        x, y,
        c=colors, s=size, marker=marker,
        edgecolors="none" if edgecolor is None else edgecolor,
        linewidths=linewidth, linestyles=linestyle, clip_on=False,
    )
    ax.set(xlim=(-0.5, width - 0.5), ylim=(-0.5, height - 0.5), aspect="equal")
    ax.set_axis_off()
    if names is not None:
        # Marker boundary, padding direction, and text alignment.
        box = points.get_paths()[0].get_extents()
        positions = {
            "top":    ((0, box.y1), (0, 1), "center", "bottom"),
            "bottom": ((0, box.y0), (0, -1), "center", "top"),
            "left":   ((box.x0, 0), (-1, 0), "right", "center"),
            "right":  ((box.x1, 0), (1, 0), "left", "center"),
            "center": ((0, 0), (0, 0), "center", "center"),
        }
        if names_pos not in positions:
            raise ValueError("names_pos must be 'top', 'bottom', 'left', 'right', or 'center'.")
        edge, direction, ha, va = positions[names_pos]
        scales = np.broadcast_to(np.sqrt(size), (n,))[:, None]
        offsets = scales * edge + (linewidth / 2 + names_pad) * np.asarray(direction)
        for xi, yi, name, offset in zip(x, y, names, offsets):
            ax.annotate(
                str(name), (xi, yi), xytext=offset,
                textcoords="offset points", ha=ha, va=va,
                fontsize=names_size,
            )
    if pl is None:
        fig.tight_layout(pad=0.4)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight", pad_inches=0.05,
                    transparent=transparent)
    if show:
        plt.show()
    return fig, ax

