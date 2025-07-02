'''
**Author**: Benjamin Urben<br>
**Email**: burben@student.ethz.ch / benjamin.urben@hotmail.ch<br>
**Context**: Master Thesis on "Use of Machine Learning in the Design and Analysis for Steel Connections"<br>
**Institution**: ETH Zürich, Institute of Structural Engineering (IBK)
'''

import os
import matplotlib as mpl
import sys
import matplotlib.font_manager as fm

class Directory:

    def __init__(self,current_folder=None):

        self.path = os.path.abspath(__file__)
        self.root = os.path.dirname(self.path)

        self.savemodel_path = os.path.join(self.root, "Saved Models")
        self.create_folder("Saved Models")

        self.savevar_path = os.path.join(self.root, "Saved Variables")
        self.create_folder("Saved Variables")
        self.illustration_path = os.path.join(self.root, "Illustrations")
        self.create_folder("Illustrations")

        if current_folder is not None:
            self.root = os.path.join(self.root,current_folder)

        os.chdir(self.root)
        print('Root Directory set to: ',self.root)

    def get_info(self):
        print('Models available to load:')
        for file in os.listdir(self.savemodel_path):
            print(file)

    def reset_root(self):
        os.chdir(self.root)
        print('Root Directory set to: ',self.root)

    def create_folder(self, folder_name):
        if not os.path.exists(os.path.join(self.root, folder_name)):
            os.mkdir(os.path.join(self.root, folder_name))
            print(f'Created folder: {folder_name}')

class Plotting_Parameters:

    def __init__(self):
        self.params = {
            'font.family': 'Charis SIL',
            'mathtext.fontset': 'stix',
            #'mathtext.fontset': 'custom',
            #'mathtext.rm': 'Charis SIL',
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'axes.linewidth': 1,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 4,
            'ytick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.minor.size': 2,
            'legend.fontsize': 9,
            'axes.linewidth': 0.75,
            'legend.frameon': False,
            'legend.loc': 'best',
            'figure.figsize': (6, 4),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'savefig.bbox': 'tight'
        }
        self.update()

        self.save_figures = False

        self.cmap = 'Set3'
        self.colors = mpl.pyplot.cm.tab10.colors
        self.morecolors = mpl.pyplot.cm.tab20.colors
        self.pastell = mpl.pyplot.cm.Set3.colors

        self.feature_labels = {
            'A_x': r'$\mathrm{A_x\ [mm^2]}$',
            'Iy_x': r'$\mathrm{I_{y,x}\ [mm^4]}$',
            'tw_x': r'$\mathrm{t_{w,x}\ [mm]}$',
            'Steel grade_x': r'$\mathrm{f_{y,x}\ [MPa]}$',
            'M_Rd': r'$\mathrm{M_{Rk}\ [kNm]}$',
            'V_Rd': r'$\mathrm{V_{Rk}\ [kN]}$',
            'd_wid': r'$\mathrm{d_{wid}\ [mm]}$',
            't_stiffc': r'$\mathrm{t_{stiff,c}\ [mm]}$',
            'h_wid': r'$\mathrm{h_{wid}\ [mm]}$',
            'Offset': r'$\mathrm{Offset\ [mm]}$',
            'h_x': r'$\mathrm{h_x\ [mm]}$',
            'h_y': r'$\mathrm{h_y\ [mm]}$',
            'b_x': r'$\mathrm{b_x\ [mm]}$',
            'b_y': r'$\mathrm{b_y\ [mm]}$',
            'A_y': r'$\mathrm{A_y\ [mm^2]}$',
            'Iy_y': r'$\mathrm{I_{y,y}\ [mm^4]}$',
            'tw_y': r'$\mathrm{t_{w,y}\ [mm]}$',
            'Steel grade_y': r'$\mathrm{f_{y,y}\ [MPa]}$',
            't_stiffb': r'$\mathrm{t_{stiff,b}\ [mm]}$',
            'tf_x': r'$\mathrm{t_{f,x}\ [mm]}$',
            'tf_y': r'$\mathrm{t_{f,y}\ [mm]}$',
            'Av_x': r'$\mathrm{A_{v,x}\ [mm^2]}$',
            'Av_y': r'$\mathrm{A_{v,y}\ [mm^2]}$',
            'Wply_y': r'$\mathrm{W_{ply,y}\ [mm^3]}$',
            'Wply_x': r'$\mathrm{W_{ply,x}\ [mm^3]}$',
            'tau_x': r'$\mathrm{\tau_x\ [MPa]}$',
            'tau_y': r'$\mathrm{\tau_y\ [MPa]}$',
            'Mpl_x': r'$\mathrm{M_{pl,x}\ [kNm]}$',
            'Mpl_y': r'$\mathrm{M_{pl,y}\ [kNm]}$',
            'Vpl_y': r'$\mathrm{V_{pl,y}\ [kN]}$',
            'Vpl_x': r'$\mathrm{V_{pl,x}\ [kN]}$',
            't_wwid': r'$\mathrm{t_{wwid}\ [mm]}$',
            't_fwid': r'$\mathrm{t_{fwid}\ [mm]}$',
            'Cat_h': r'$\mathrm{Cat_h\ [mm]}$',
            'Cat_t_stiffc': r'$\mathrm{Cat_{t_{stiff,c}}\ [mm]}$',
            'b_wid': r'$\mathrm{b_{wid}\ [mm]}$',
            'M': r'$\mathrm{M\ [kNm]}$',
            'V': r'$\mathrm{V\ [kN]}$',
            'fy_x': r'$\mathrm{f_{y,x}\ [MPa]}$',
            'fy_y': r'$\mathrm{f_{y,y}\ [MPa]}$',
            'Gamma': r'$\mathrm{\gamma\ [\degree]}$',
        }

    def repeated_colors(self, values):
        """Return a list of colors from the colormap, repeated if necessary."""
        return [self.morecolors[int(value) % len(self.morecolors)] for value in values]
    
    def update(self):
        mpl.rcParams.update(self.params)

    def get_available_fonts(self):
        # Get a list of all font objects known to Matplotlib
        font_list = fm.fontManager.ttflist + fm.fontManager.afmlist

        # Extract unique font family names
        available_font_families = sorted(list(set([f.name for f in font_list])))

        print("Available Font Families in Matplotlib:")
        for font_family in available_font_families:
            print(f"- {font_family}")

    def get_figsize(self,fraction,aspect_ratio=None, document_width=15.92):
        width_in = document_width / 2.54 * fraction

        if aspect_ratio is None:
            aspect_ratio = (5 ** 0.5 - 1) / 2  # golden ratio

        height_in = width_in * aspect_ratio
        return width_in, height_in
