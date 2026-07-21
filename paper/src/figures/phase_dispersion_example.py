# Imports 
import numpy as np, bruce
np.random.seed(0)
import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
import matplotlib
from scipy.signal import find_peaks

# Example usage
night = np.arange(-0.25,0.25, 5/60/24).astype(np.float64)  # Example x-values (e.g., time) as double

x = night.copy()
for i in range(1,101) : x = np.concatenate((x,night+i))
for i in range(150+1,150+101) : x = np.concatenate((x,night+i))
    
segments = bruce.data.find_nights_from_data(x, dx_lim=10)    
    
    
# Transit parameters
transit_parameters = {}
transit_parameters['t_zero'] = 12.3
transit_parameters['period'] = 21.4
transit_parameters['radius_1'] = 0.08
transit_parameters['k'] = 0.08
transit_parameters['incl'] = np.arccos(transit_parameters['radius_1']*0.4)

transit_parameters_ = transit_parameters.copy()
transit_parameters__ = transit_parameters.copy()

del transit_parameters_['t_zero']
del transit_parameters__['t_zero']
del transit_parameters__['period']


Noise = 0.01
ye = np.random.uniform(0.5*Noise,1.5*Noise, x.shape)
y = 1 + 0.04*np.sin(0.4*np.pi + 2*np.pi*x/45) + 0.005*np.sin(2*np.pi*x/67) + 0.01*np.cos(0.9*np.pi + 2*np.pi*x/67) + 0.005*np.sin(2*np.pi*x/67) + np.random.normal(np.zeros(x.shape), ye)       # Example y-values (e.g., signal values) as double
y = y*bruce.binarystar.lc(x, **transit_parameters)

y_filtered =  bruce.data.median_filter(x,y, bin_size=4)
y_filtered = bruce.data.convolve_1d(x,y_filtered, bin_size=2)


time_trial, DeltaL = bruce.template_match.template_match_lightcurve(x,y,ye,y_filtered,**transit_parameters_ )


# Calibrated per-epoch heights:
_, rho2 = bruce.template_match.template_rho2(x, ye, y_filtered,
                                             time_trial=time_trial,
                                             **transit_parameters_)
pvalues, heights = bruce.template_match.get_delta_loglike_height_from_fap(rho2=rho2)

peaks, meta = find_peaks(DeltaL, height=heights[2], distance=100)


mosaic = '''AAA
BBB
CCC
DEF
GHI
'''
props = dict(boxstyle='round', facecolor='wheat', alpha=1)

fig, ax = plt.subplot_mosaic(mosaic,figsize=(7,7))
ax = [ax['A'],ax['B'],ax['C'], ax['D'],ax['E'],ax['F'], ax['G'],ax['H'], ax['I']]
ax[0].scatter(x,y, c='k',s=1, alpha=0.1)
for seg in segments: ax[0].plot(x[seg],y_filtered[seg],c='b',lw=1)
ax[0].set(ylabel='Flux', xlim=(time_trial[0]-3, time_trial[-1]+3), xticks=[])

# Now lets mark where the transits are
width = bruce.binarystar.transit_width(transit_parameters['radius_1'], transit_parameters['k'],0.4, transit_parameters['period'])
phase_width = width / transit_parameters['period']
phase_of_data = bruce.data.phase_times(x, transit_parameters['t_zero'], transit_parameters['period'], phase_offset=0.2)
x_in_transit = x[(phase_of_data>(-phase_width/2)) & (phase_of_data<(phase_width/2))]
y_in_transit = (y/y_filtered)[(phase_of_data>(-phase_width/2)) & (phase_of_data<(phase_width/2))]

in_transit_segments = bruce.data.find_nights_from_data(x_in_transit, dx_lim=0.2)
cycles = np.array([int(np.round((x_in_transit[seg[0]] - transit_parameters['t_zero']) /transit_parameters['period'])) for seg in in_transit_segments])
counts = np.array([len(i) for i in in_transit_segments])

cycles = cycles[counts>10]

t_zeros = [transit_parameters['t_zero'] + transit_parameters['period']*i for i in cycles]
# print(cycles)
# print(counts)
# print(t_zeros)


# plt.close()
# fig, ax = plt.subplots(4,3)
# ax = np.ndarray.flatten(ax)
# for i in range(len(in_transit_segments)):
#     ax[i].scatter(x_in_transit[in_transit_segments[i]], y_in_transit[in_transit_segments[i]], c='k', s=1)
#     t_ = np.linspace(t_zeros[i]-width, t_zeros[i]+width, 1000)
#     model = bruce.binarystar.lc(t_, t_zero=t_zeros[i], **transit_parameters_)
#     ax[i].plot(t_, model, c='orange', lw=1, alpha=1)
# plt.show()


for i in range(len(t_zeros)):
    ax[0].plot(t_zeros[i], 1.1, marker=7, color='b')


ax[1].plot(time_trial, DeltaL, c='k',lw=1)
for i in range(len(peaks)):
    #ax[1].plot(time_trial[peaks[i]], DeltaL[peaks[i]]+3, c='r', marker=7)
    #ax[1].text(time_trial[peaks[i]], DeltaL[peaks[i]]+5, str(i+1), va='bottom', ha='center')
    ax[1].text(time_trial[peaks[i]], DeltaL[peaks[i]]+5, str(i+1), fontsize=10,
            verticalalignment='bottom', horizontalalignment='center', bbox=props)
ax[1].set(xlabel='Time [day]', ylabel=r'$\Delta \log \mathcal{L}$', ylim=(0,72),  xlim=(time_trial[0]-3, time_trial[-1]+3))

ax[1].plot(time_trial, heights[0], c='k', ls='dashed', lw=1, label='1%')
ax[1].plot(time_trial, heights[1], c='k', ls='dotted', lw=1, label='0.1%')
ax[1].plot(time_trial, heights[2], c='k', ls='dashdot', lw=1, label='0.01%')
ax[1].legend(fontsize=7)

# Now lets do the phase dispersion
period, sum_of_deltas, chi_squared = bruce.template_match.phase_disperison(time_trial, peaks, x,y/y_filtered,ye/y_filtered,
                                                                           minimum_period=1, maximum_period=100, samples_per_peak=100)

ax[2].loglog(period, sum_of_deltas, c='k', lw=1)
ax[2].plot(transit_parameters['period'], 5, marker=7, color='r')
for i in range(2,22) : ax[2].plot(transit_parameters['period']/i, 6 if (((i%2)==0) and (transit_parameters['period']/i)<2) else 5, marker=7, color='b')


ax[2].set(ylabel = r'PD', xlabel='Period [day]')
ax[2].set_xticks([1,10,100], ['1','10','100'])
ax[2].set_yticks([1,10], ['1','10'])

def move_axis(ax, dy, left=None):

    pos = ax.get_position()
    pos.y0 = pos.y0 +dy     # for example 0.2, choose your value
    pos.y1 = pos.y1 +dy     # for example 0.2, choose your value
    if left is not None:
        pos.x0 = left[0]
        pos.x1 = left[1]
    ax.set_position(pos)

left = [0.09,0.99]
move_axis(ax[0], dy=0.11, left = left)
move_axis(ax[1], dy=0.12, left = left)
move_axis(ax[2], dy=0.06,  left = left)

dy=-0.0
left1 = [0.12, 0.38]
left2 = [0.38+0.015, 0.685]
left3 = [0.7, 0.99]
move_axis(ax[3], dy=dy, left = left1)
move_axis(ax[4], dy=dy, left = left2)
move_axis(ax[5], dy=dy, left = left3)

dy=-0.03
move_axis(ax[6], dy=dy, left = left1)
move_axis(ax[7], dy=dy, left = left2)
move_axis(ax[8], dy=dy, left = left3)

ylim=[0.95,1.05]
for i in range(min(len(peaks), len(ax) - 3)):
    width = 1.5
    mask = (x>(time_trial[peaks[i]]-width)) & (x<(time_trial[peaks[i]]+width))
    #ax[i+3].scatter(x[mask], y[mask]/y_filtered[mask], c='k', s=1)
    ax[i+3].errorbar(x[mask], y[mask]/y_filtered[mask], yerr=ye[mask]/y_filtered[mask], fmt='k.', markersize=1, lw=1, alpha = 0.05)
    t_bin, f_bin, fe_bin, _ = bruce.data.bin_data(x[mask], y[mask]/y_filtered[mask], bin_size=0.5/24)
    ax[i+3].errorbar(t_bin, f_bin, yerr=fe_bin, fmt='b.', markersize=1, lw=1)
    ax[i+3].set_xlim(time_trial[peaks[i]]-width,time_trial[peaks[i]]+width)
    
    t = np.linspace(time_trial[peaks[i]]-width,time_trial[peaks[i]]+width, 1000)
    model = bruce.binarystar.lc(t, t_zero=time_trial[peaks[i]], **transit_parameters_)
    ax[i+3].plot(t, model, c='orange', lw=1, alpha=1)
    ax[i+3].set_ylim(ylim)
    

    # place a text box in upper left in axes coords
    ax[i+3].text(0.0, 1, str(i+1), transform=ax[i+3].transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left', bbox=props)


ax[6].set_xlabel('Time [day]')
ax[7].set_xlabel('Time [day]')
ax[8].set_xlabel('Time [day]')
ax[3].set_ylabel('Flux')
ax[6].set_ylabel('Flux')

ax[4].set_yticks([])
ax[5].set_yticks([])
ax[7].set_yticks([])

plt.savefig('phase_dispersion_example.pdf')
plt.close()


# from tqdm import tqdm 
# chis = np.zeros(len(period))
# for i in tqdm(range(len(period))[:]):
#     phases = bruce.data.phase_times(time_trial[peaks], 0, period[i])
#     phase_of_data = bruce.data.phase_times(x,0, period[i])
#     model = bruce.binarystar.lc(phase_of_data, t_zero=np.median(phases), period=1, **transit_parameters__)
#     chis[i] = -2*bruce.sampler.loglike(y/y_filtered,ye/y_filtered,model, jitter=0., offset=False)

# #plt.plot(period, sum_of_deltas)

# plt.semilogx(period, -(chis - (np.median(chis)))/np.std(chis))
# plt.show()

plt.plot(period, chi_squared)
plt.show()
