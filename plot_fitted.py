import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

if argc < 1:
  print("Usage:\t", sys.argv[0], "lcfile_fitted")
  sys.exit(1)



def binto(x=None, y=None, yuncertainty=None,
			binwidth=0.01,
			test=False,
			robust=True,
			sem=True,
			verbose=False):
	'''Bin a timeseries to a given binwidth,
		returning both the mean and standard deviation
			(or median and approximate robust scatter).'''

	if test:
		n = 1000
		x, y = np.arange(n), np.random.randn(n) - np.arange(n)*0.01 + 5
		bx, by, be = binto(x, y, binwidth=20)
		plt.figure('test of zachopy.binto')
		plt.cla()
		plt.plot(x, y, linewidth=0, markersize=4, alpha=0.3, marker='.', color='gray')
		plt.errorbar(bx, by, be, linewidth=0, elinewidth=2, capthick=2, markersize=10, alpha=0.5, marker='.', color='blue')
		return

	min, max = np.min(x), np.max(x)
	bins = np.arange(min, max+binwidth, binwidth)
	count, edges = np.histogram(x, bins=bins)
	sum, edges = np.histogram(x, bins=bins, weights=y)

	if yuncertainty is not None:
		count, edges = np.histogram(x, bins=bins)
		numerator, edges = np.histogram(x, bins=bins, weights=y/yuncertainty**2)
		denominator, edges = np.histogram(x, bins=bins, weights=1.0/yuncertainty**2)
		mean = numerator/denominator
		std = np.sqrt(1.0/denominator)
		error = std
		if False:
			for i in range(len(bins)-1):
				print(bins[i], mean[i], error[i], count[i])
			a = raw_input('???')
	else:
		if robust:
			n= len(sum)
			mean, std = np.zeros(n) + np.nan, np.zeros(n) + np.nan
			for i in range(n):
				inbin = (x>edges[i])*(x<=edges[i+1])
				mean[i] = np.median(y[inbin])
				std[i] = 1.48*mad(y[inbin])
		else:
			if yuncertainty is None:
				mean = sum.astype(np.float)/count
				sumofsquares, edges = np.histogram(x, bins=bins, weights=y**2)
				std = np.sqrt(sumofsquares.astype(np.float)/count - mean**2)*np.sqrt(count.astype(np.float)/np.maximum(count-1.0, 1.0))
		if sem:
			error = std/np.sqrt(count)
		else:
			error = std


	x = 0.5*(edges[1:] + edges[:-1])
	return x, mean, error

	if yuncertainty is not None:
		print("Uh-oh, the yuncertainty feature hasn't be finished yet.")

	if robust:
		print("Hmmm...the robust binning feature isn't finished yet.")


        
  
f = sys.argv[1]

df = pd.read_csv(f, comment='#')


with open(f) as myfile:
    head = [next(myfile) for x in range(4)]

period = float(head[0].strip('\n').split('=')[1])
amp = float(head[1].strip('\n').split('=')[1])
e_amp = float(head[2].strip('\n').split('=')[1])
t0 = float(head[3].strip('\n').split('=')[1])


modx = np.linspace(np.min(df['time']), np.max(df['time']), 1000)
mody = amp*np.sin(2*math.pi*(modx - t0)/period)


plt.plot(df['time'], df['ycorr'],'.', c='gray')
plt.plot(modx, mody)
plt.show()

