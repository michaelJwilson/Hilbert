import  matplotlib;              matplotlib.use('PDF')
import  numpy             as     np
import  pylab             as     pl
import  matplotlib.pyplot as     plt

plt.style.use('ggplot')

## import  matplotlib.image  as     mpimg
## from    PIL               import Image
## from    skimage           import data, color, io
## from    skimage.transform import rescale, resize, downscale_local_mean

fig   = plt.figure()

x     = np.arange(0, 100,  1)
y     = 5. * x

x1max = x[-1]
y1max = y[-1]

pl.plot(x, y, 'k')

x     = np.arange(0, -100, -1)
y     = -4. * x 

x2max = x[-1]
y2max = y[-1]

pl.plot(x, y, 'k')
pl.plot([x1max, x2max], [y1max, y2max], 'k')


x     = np.arange(0, 200,  1)
y     = 1. * x

x3max = x[-1]
y3max = y[-1]

pl.plot(x, y, 'k')

x     = np.arange(0, 200, 1)
y     = -1. * x

x4max = x[-1]
y4max = y[-1]

pl.plot(x, y, 'k')

pl.plot(np.array([x3max, x4max]), np.array([y3max, y4max]), 'k')

pl.plot(np.array([x2max, x3max]), np.array([y2max, y3max]), 'k--', alpha=0.3)
pl.plot(np.array([x1max, x4max]), np.array([y1max, y4max]), 'k--', alpha=0.3)

## pl.arrow(0.0, 0.0, x1max, y1max, fc="k", ec="k", head_width=10., head_length=10.)

ax  = pl.gca()

for (x, y, label) in [(x1max, y1max, r"$\vec{s}_1'$"), (x2max, y2max, r'$\vec{s}_1$'), (x3max, y3max, r'$\vec{s}_2$'), (x4max, y4max, r"$\vec{s}_2'$")]:
    ax.annotate(r'', xycoords='data', xytext = (0.45 * x, 0.45 * y), xy=(0.55 * x, 0.55 * y), \
                     arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='bottom')

    ax.annotate(label, xy= (0.65 * x, 0.5 * y), xytext=(0.65 * x, 0.5 * y), \
                xycoords='data', horizontalalignment='center', verticalalignment='center')

##ax.annotate(r'', xycoords='data', xytext = (0.4 * (x1max + x2max), 0.4 * (y1max + y2max)), xy=(0.6 * (x1max + x2max), 0.6 * (y1max + y2max)), \
##                 arrowprops=dict(facecolor='black', shrink=0.05), horizontalalignment='left', verticalalignment='bottom')

plt.axis('off')

pl.savefig('triangle.pdf', bbox_inches='tight')
