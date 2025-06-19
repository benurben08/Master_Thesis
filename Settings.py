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
            'font.size': 11,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'axes.linewidth': 1,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'xtick.major.size': 5,
            'ytick.major.size': 5,
            'xtick.minor.size': 3,
            'ytick.minor.size': 3,
            'legend.fontsize': 9,
            'legend.frameon': False,
            'legend.loc': 'best',
            'figure.figsize': (6, 4),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.format': 'png',
            'savefig.bbox': 'tight'
        }
        self.update()

        self.cmap = 'Set3'
        self.colors = mpl.pyplot.cm.tab10.colors
        self.morecolors = mpl.pyplot.cm.tab20.colors
        self.pastell = mpl.pyplot.cm.Set3.colors

        self.feature_labels = {'A_x': r'$A_x$ $\mathrm{[mm^2]}$',
                  'Iy_x': r'$I_{y,x}$ $\mathrm{[mm^4]}$',
                  'tw_x': r'$t_{w,x}$ $\mathrm{[mm]}$',
                  'Steel grade_x': r'$f_{y,x}$ $\mathrm{[MPa]}$',
                  'M_Rd': r'$M_{Rd}$ $\mathrm{[kNm]}$',
                  'V_Rd': r'$V_{Rd}$ $\mathrm{[kN]}$',
                  'd_wid': r'$d_{wid}$ $\mathrm{[mm]}$',
                  't_stiffc': r'$t_{stiff,c}$ $\mathrm{[mm]}$',
                  'h_wid': r'$h_{wid}$ $\mathrm{[mm]}$',
                  'Offset': r'$Offset$ $\mathrm{[mm]}$',
                  'h_x': r'$h_x$ $\mathrm{[mm]}$',
                  'h_y': r'$h_y$ $\mathrm{[mm]}$',
                  'b_x': r'$b_x$ $\mathrm{[mm]}$',
                  'b_y': r'$b_y$ $\mathrm{[mm]}$',
                  'A_y': r'$A_y$ $\mathrm{[mm^2]}$',
                  'Iy_y': r'$I_{y,y}$ $\mathrm{[mm^4]}$',
                  'tw_y': r'$t_{w,y}$ $\mathrm{[mm]}$',
                  'Steel grade_y': r'$f_{y,y}$ $\mathrm{[MPa]}$',
                  't_stiffb': r'$t_{stiff,b}$ $\mathrm{[mm]}$',
                  'tf_x': r'$t_{f,x}$ $\mathrm{[mm]}$',
                  'tf_y': r'$t_{f,y}$ $\mathrm{[mm]}$',
                  'Av_x': r'$A_{v,x}$ $\mathrm{[mm^2]}$',
                  'Av_y': r'$A_{v,y}$ $\mathrm{[mm^2]}$',
                  'Wply_y': r'$W_{ply,y}$ $\mathrm{[mm^3]}$',
                  'Wply_x': r'$W_{ply,x}$ $\mathrm{[mm^3]}$',
                  'tau_x': r'$\tau_{x}$ $\mathrm{[MPa]}$',
                  'tau_y': r'$\tau_{y}$ $\mathrm{[MPa]}$',
                  'Mpl_x': r'$M_{pl,x}$ $\mathrm{[kNm]}$',
                  'Mpl_y': r'$M_{pl,y}$ $\mathrm{[kNm]}$',
                  'Vpl_y': r'$V_{pl,y}$ $\mathrm{[kN]}$',
                  'Vpl_x': r'$V_{pl,x}$ $\mathrm{[kN]}$',
                  't_wwid': r'$t_{wwid}$ $\mathrm{[mm]}$',
                  't_fwid': r'$t_{fwid}$ $\mathrm{[mm]}$',
                  'Cat_h': r'$Cat_{h}$ $\mathrm{[mm]}$',
                  'Cat_t_stiffc': r'$Cat_{t_{stiff,c}}$ $\mathrm{[mm]}$',
                  'b_wid': r'$b_{wid}$ $\mathrm{[mm]}$',
                  'M': r'$M$ $\mathrm{[kNm]}$',
                  'V': r'$V$ $\mathrm{[kN]}$',
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
