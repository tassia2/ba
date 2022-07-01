import numpoly 
import numpy as np
import matplotlib.pyplot as plt
import os
from configparser import ConfigParser
import re 
from matplotlib.widgets import Slider, Button

class Plotter:
    def __init__(self,polyfolder) -> None:
        # config = ConfigParser()
        # config.read(PC_config_path)
        # dist_str = config.get('main', 'dist_list')
        # # dist_array = dist_str[1:-1].split(",")
        # self.dist_boundaries = re.findall("[\d,.]+",dist_str)
        x = self.polymatrix
        for i in range(65):
            x.append([])
            for j in range(65):
                x[i].append(numpoly.load(os.path.join(polyfolder,str(i*65+j)+".npy")))

    def static_plot(self):
        fig, ax = plt.subplots()
        init_prandtl = 5
        init_rayleigh = 1e5
        # Make data.
        # rng = list(range(65))
        # X, Y = np.meshgrid(rng,rng)

        def f(prandtl,rayleigh):
            Z = np.empty((65,65))
            for i in range(65):
                for j in range(65):
                    Z[i,j] = self.polymatrix[i][j](prandtl, rayleigh)
            return Z

        # Y, X = np.mgrid[-3:3+0.1:0.1, -3:3+0.1:0.1]
        # Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
        # Z = Z[:-1, :-1]
        # z_min, z_max = -abs(Z).max(), abs(Z).max()
        
        # c = plt.pcolor(X, Y, f(init_prandtl,init_rayleigh), cmap='RdBu')
        c = plt.imshow(f(init_prandtl,init_rayleigh), cmap='RdBu',
              interpolation='nearest', origin='lower', aspect='auto', vmax=1,vmin=0)
            #   vmin=z_min, vmax=z_max,extent=[x.min(), x.max(), y.min(), y.max()],

        fig.colorbar(c)
        ax.set_title("Pressure - Timestep 50")
        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(bottom=0.25)
        # Make a horizontal slider to control the frequency.
        axfreq = plt.axes([0.15, 0.1, 0.6, 0.03])
        prandtl_slider = Slider(
            ax=axfreq,
            label='Prandtl Number',
            valmin=2,
            valmax=13.5,
            valinit=init_prandtl,
        )

        # Make a vertically oriented slider to control the amplitude
        axamp = plt.axes([0.15, 0.13, 0.6, 0.03])
        rayleigh_slider = Slider(
            ax=axamp,
            label="Rayleigh Number",
            valmin=1e4,
            valmax=1e6,
            valinit=init_rayleigh
        )

        # The function to be called anytime a slider's value changes
        def update(val):
            # c.set_ydata(f( prandtl_slider.val, rayleigh_slider.val))
            c.set_data(f( prandtl_slider.val, rayleigh_slider.val)) 
            fig.canvas.draw_idle()

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Set', hovercolor='0.975')

        def SET(event):
            data = f( prandtl_slider.val, rayleigh_slider.val)
            c.set_data(data)
            # c.set_clim([np.min(data), np.max(data)]) 
            fig.canvas.draw_idle()

        button.on_clicked(SET)

        # register the update function with each slider
        # rayleigh_slider.on_changed(update)
        # prandtl_slider.on_changed(update)

        plt.show()
        print("PLOTTET ")

    polymatrix = []


polyfolder = /home/tassia/Hiflow-2-22/ba-v2/build/examples/boussinesq2d
Plot = Plotter(polyfolder)
Plot.static_plot()
