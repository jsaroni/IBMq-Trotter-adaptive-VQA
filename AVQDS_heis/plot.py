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

nsite = 4


"""
commented out
with h5py.File("../4q/T3/results_exact.h5", "r") as f:
with h5py.File("results_exact_4q.h5", "r") as f:
   tov_0 = [f["/0/t"][()], f["/0/ov"][()].T]
    te_0 = [f["/0/t"][()], f["/0/e"][()]]
"""



with h5py.File("results_asim.h5", "r") as f:
    tov_1 = [f["/0/t"][()], f["/0/ov"][()].T]
    te_1 = [f["/0/t"][()], f["/0/e"][()]]
    tmag_s_1 = [f["/0/t"][()], f["/0/mag_s"][()].T]
    tgates_1 = [f["/0/t"][()], f["/0/ngates"][()].T]
    
    
    

lamp = numpy.abs(tov_1[1])**2


print(lamp)





    
    
    

#lamb_list_0 = numpy.minimum(-numpy.log(tov_0[1][0, :]), -numpy.log(tov_0[1][1, :]))/nsite*2
#lamb_list_1 = numpy.minimum(-numpy.log(tov_1[1][0, :]), -numpy.log(tov_1[1][1, :]))/nsite*2


#plt.figure(0)
# plt.plot(te_0[0], te_0[1], "o")
# plt.plot(te_1[0], te_1[1], "-")
# plt.figure(1)
# plt.plot(tov_0[0], lamb_list_0, ":")
# plt.plot(tov_1[0], lamb_list_1, "-")
# plt.figure(2)
# plt.plot(te_1[0], tgates_1[1][0], "-")
# plt.plot(te_1[0], tgates_1[1][1], "-")
# plt.show()


#+ 0.258065*tmag_s_1_15_14[1] + 0.193548*tmag_s_1_7_15[1] + 0.193548*tmag_s_1_13_14[1] + 0.16129*tmag_s_1_7_7[1] 


#print(tmag_s_1_15_15[1])

fig = plt.figure(1,figsize = [width,height])
ax1 = plt.subplot(1,1,1)
ax1.plot(tov_1[0], lamp, marker='o', color=color_red)
ax1.grid(linestyle=':', linewidth=1.)
ax1.set_xlabel("$t/J$")
ax1.set_ylabel("$m_s$")
ax1.set_title(r"Initial state: see run.py, Final H: see run.py $h_x = -2J$, $h_z=0.5J$")
#ax1.yscale("log")

plt.tight_layout()

fig.savefig("Plot-Magnetization_staggered.pdf", bbox_inches="tight")

plt.show()

plt.clf()
plt.cla()
plt.close()



#print( tmag_s_1[1])

x = numpy.array(tmag_s_1[0])

y = numpy.array(tmag_s_1[1])






#plt.plot(x, y)



"""
fig2 = plt.figure(1,figsize = [width,height])
ax2 = plt.subplot(1,1,1)
ax2.plot(tov_1[0], tov_1[1][0,:], marker='o', color=color_red)
ax2.grid(linestyle=':', linewidth=1.)
ax2.set_xlabel("$t/J$")
ax2.set_ylabel("Overlap with initial state")
ax2.set_title(r"Initial state: see run.py, Final H: see run.py $h_x = -2J$, $h_z=0.5J$")
#ax1.yscale("log")

plt.tight_layout()

fig2.savefig("Plot-Overlap_with_initial_state.pdf", bbox_inches="tight")

plt.show()

"""