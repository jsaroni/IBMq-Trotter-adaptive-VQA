import h5py, numpy
import matplotlib.pyplot as plt
from matplotlib import rc


rc('font', **{'family':'sans-serif', 'size' : 12})
rc('text', usetex=True)

plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)

# define colors
color_red = (0.73, 0.13869999999999993, 0.)
color_orange = (1., 0.6699999999999999, 0.)
color_green = (0.14959999999999996, 0.43999999999999995, 0.12759999999999994)
color_blue = (0.06673600000000002, 0.164512, 0.776)
color_purple = (0.25091600000000003, 0.137378, 0.29800000000000004)
color_ocker = (0.6631400000000001, 0.71, 0.1491)
color_pink = (0.71, 0.1491, 0.44730000000000003)
color_brown = (0.651, 0.33331200000000005, 0.054683999999999955)
color_red2 = (0.766, 0.070, 0.183)
color_turquoise = (0., 0.684, 0.676)
color_yellow = (0.828, 0.688, 0.016)
color_grey = (0.504, 0.457, 0.410)
width = 6
height = width / 1.618

nsite = 3




with h5py.File("results_asim.h5", "r") as f:
    tov = [f["/0/t"][()], f["/0/ov"][()].T]
    te = [f["/0/t"][()], f["/0/e"][()]]
    tmag_s = [f["/0/t"][()], f["/0/mag_s"][()].T]
    tgates = [f["/0/t"][()], f["/0/ngates"][()].T]
    
    
    

Lecho = numpy.abs(tov[1])**2    
    

fig = plt.figure(1,figsize = [width,height])
ax1 = plt.subplot(1,1,1)
ax1.plot(tov[0], Lecho, marker='o', color=color_red)
ax1.grid(linestyle=':', linewidth=1.)
ax1.set_xlabel("t")
ax1.set_ylabel("Loschmidt echo")
ax1.set_title(r"Initial state: 110")
plt.tight_layout()

fig.savefig("Loschmidt echo.pdf", bbox_inches="tight")

plt.show()

plt.clf()
plt.cla()
plt.close()

