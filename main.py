# This code takes in a set of [x,y] points with error [dx,dy] and fits a line to the data

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# The Number of Monte-Carlo Sample Points
Nsample = 500

# How many times will we run the Program an generate Samples?
NbootStraps = 200

# Read in the [x0_i,dx_i,y0_i,dy_i] data points (This is sample data for the moment)
x_axis_name = 'X-data'
y_axis_name = 'Y-data'
font_size = 20

# Replace this data with your own stuff:
x = [0.0,1.0,2.0,3.0,4.0,5.0]
dx = [0.01,0.01,0.01,0.01,0.01,0.01]

y = [1.0,2.0,3.0,4.0,5.0,6.0]
dy = [0.003,0.1,0.3,0.1,0.1,1.0]


#==================================================================================================

# These arrays store all results of the bootstrap samples
slope_samples = []
sigma_slope_samples = []

intercept_samples = []
sigma_intercept_samples = []

for b in range(NbootStraps):
    #=================================================================================
    # Create N samples of the vector x = [x0_1,...,x0_M], y = [y0_1,...,y0_M]
    slope_pop = []
    intercept_pop = []
    r_value_pop = []
    
    for k in range(Nsample):
        xk = np.random.normal(loc=x, scale=dx)
        yk = np.random.normal(loc=y, scale=dy)
        slope_k, intercept_k, r_value_k, p_value, std_err = stats.linregress(xk,yk)
        slope_pop.append(slope_k)
        intercept_pop.append(intercept_k)
        r_value_pop.append(r_value_k)
    #=================================================================================
    
    # Compute the Mean and Variance of the Population
    mu_slope =  np.mean(slope_pop, dtype=np.float64)
    sigma_slope =  np.std(slope_pop, dtype=np.float64)
    
    mu_intercept =  np.mean(intercept_pop, dtype=np.float64)
    sigma_intercept =  np.std(intercept_pop, dtype=np.float64)
    
    # Append the data to the bootstrap Sample
    slope_samples.append(mu_slope)
    sigma_slope_samples.append(sigma_slope)
    
    intercept_samples.append(mu_intercept)
    sigma_intercept_samples.append(sigma_intercept)


#================================================================================
# Compute the statistics of the Bootstrap Samples
#================================================================================

slope_mean = np.mean(slope_samples,dtype=np.float64)
sigma_slope_mean = np.mean(sigma_slope,dtype=np.float64)

intercept_mean =  np.mean(intercept_samples,dtype=np.float64)
sigma_intercept_mean =  np.mean(sigma_intercept_samples,dtype=np.float64)

print '======================================================================='
print 'slope: ', slope_mean
print 'sigma_slope: ', sigma_slope_mean
print '------------------------------------------------------------------'
print 'The Following data gives an error estimate of the above parameters (smaller the better)'
print 'Variance of slope: ', np.std(slope_samples,dtype=np.float64)
print 'Variance of slope sigma: ', np.std(sigma_slope_samples,dtype=np.float64)
print '========================================================================'

print '========================================================================'
print 'intercept: ', intercept_mean 
print 'sigma_intercept: ', sigma_intercept_mean
print '---------------------------'
print 'The Following data gives an error estimate of the above parameters (smaller the better)'
print 'Variance of intercept: ', np.std(intercept_samples,dtype=np.float64)
print 'Variance of intercept sigma: ', np.std(sigma_intercept_samples,dtype=np.float64)
print '========================================================================='

#================================================================================

# Now we plot the results
x = np.asarray(x)
plt.xlabel(x_axis_name,size=font_size)
plt.ylabel(y_axis_name,size=font_size)
line = 'Slope: '+ str(slope_mean)+' +/- '+str(sigma_slope_mean)+'\n'
line2 = 'Intercept: '+ str(intercept_mean)+' +/- '+str(sigma_intercept_mean)
plt.title(line+'\n'+line2)
plt.errorbar(x, y, xerr=dx, yerr=dy)
plt.plot(x,slope_mean*x+intercept_mean, color = 'blue' )
plt.plot(x,(slope_mean+sigma_slope_mean)*x+intercept_mean+sigma_intercept_mean,color = 'green')
plt.plot(x,(slope_mean-sigma_slope_mean)*x+intercept_mean-sigma_intercept_mean,color = 'green')
plt.fill_between(x,(slope_mean-sigma_slope_mean)*x+intercept_mean-sigma_intercept_mean,(slope_mean+sigma_slope_mean)*x+intercept_mean+sigma_intercept_mean,color='green',alpha='0.6')
plt.savefig("line.pdf", bbox_inches='tight')
plt.show()




