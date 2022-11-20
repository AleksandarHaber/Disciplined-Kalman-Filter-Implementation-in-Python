# -*- coding: utf-8 -*-
"""
Kalman filter implementation in Python

Driver code for testing the class KalmanFilter that implements the Kalman filter equations.

The tutorial explaining this code is given here:

https://aleksandarhaber.com/disciplined-kalman-filter-implementation-in-python-by-using-object-oriented-approach/

Created by: Aleksandar Haber
Last edited: November 20, 2022    

"""

import numpy as np
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter


# discretization step 
h=0.1
# initial values for the simulation
initialPosition=10
initialVelocity=-5
# acceleration
acceleration=0.5
# measurement noise standard deviation
noiseStd=1;
# number of discretization time steps
numberTimeSteps=100

# define the system matrices - Newtonian system
# system matrices and covariances
A=np.matrix([[1,h,0.5*(h**2)],[0, 1, h],[0,0,1]])
B=np.matrix([[0],[0],[0]])
C=np.matrix([[1,0,0]])

R=1*np.matrix([[1]])
Q=np.matrix(np.zeros((3,3)))

#guess of the initial estimate
x0=np.matrix([[0],[0],[0]])
# initial covariance matrix 
P0=1*np.matrix(np.eye(3))

# time vector for simulation
timeVector=np.linspace(0,(numberTimeSteps-1)*h,numberTimeSteps)

# vector used to store the simulated position
position=np.zeros(np.size(timeVector))
velocity=np.zeros(np.size(timeVector))

# simulate the system behavior
for i in np.arange(np.size(timeVector)):
    position[i]=initialPosition+initialVelocity*timeVector[i]+(acceleration*timeVector[i]**2)/2
    velocity[i]=initialVelocity+acceleration*timeVector[i]
    
# add the measurement noise 
positionNoisy=position+noiseStd*np.random.randn(np.size(timeVector))

# verify the position vector by plotting the results
plotStep=numberTimeSteps//2
plt.plot(timeVector[0:plotStep],position[0:plotStep],linewidth=4, label='Ideal position')
plt.plot(timeVector[0:plotStep],positionNoisy[0:plotStep],'r', label='Observed position')
plt.xlabel('time')
plt.ylabel('position')
plt.legend()
plt.savefig('data.png',dpi=300)
plt.show()

#create a Kalman filter object 
KalmanFilterObject=KalmanFilter(x0,P0,A,B,C,Q,R)
inputValue=np.matrix([[0]])
# simulate online prediction
for j in np.arange(np.size(timeVector)):
    KalmanFilterObject.propagateDynamics(inputValue)
    KalmanFilterObject.computeAposterioriEstimate(positionNoisy[j])

KalmanFilterObject.estimates_aposteriori

# extract the state estimates in order to plot the results
estimate1=[]
estimate2=[]
estimate3=[]    
for j in np.arange(np.size(timeVector)):
    estimate1.append(KalmanFilterObject.estimates_aposteriori[j][0,0])
    estimate2.append(KalmanFilterObject.estimates_aposteriori[j][1,0])
    estimate3.append(KalmanFilterObject.estimates_aposteriori[j][2,0])
    
# create vectors corresponding to the true values in order to plot the results
estimate1true=position
estimate2true=velocity
estimate3true=acceleration*np.ones(np.size(timeVector))


# plot the results
steps=np.arange(np.size(timeVector))
fig, ax = plt.subplots(3,1,figsize=(10,15))
ax[0].plot(steps,estimate1true,color='red',linestyle='-',linewidth=6,label='True value of position')
ax[0].plot(steps,estimate1,color='blue',linestyle='-',linewidth=3,label='Estimate of position')
ax[0].set_xlabel("Discrete-time steps k",fontsize=14)
ax[0].set_ylabel("Position",fontsize=14)
ax[0].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[0].set_ylim(98,102)  
ax[0].grid()
ax[0].legend(fontsize=14)

ax[1].plot(steps,estimate2true,color='red',linestyle='-',linewidth=6,label='True value of velocity')
ax[1].plot(steps,estimate2,color='blue',linestyle='-',linewidth=3,label='Estimate of velocity')
ax[1].set_xlabel("Discrete-time steps k",fontsize=14)
ax[1].set_ylabel("Velocity",fontsize=14)
ax[1].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[1].grid()
ax[1].legend(fontsize=14)

ax[2].plot(steps,estimate3true,color='red',linestyle='-',linewidth=6,label='True value of acceleration')
ax[2].plot(steps,estimate3,color='blue',linestyle='-',linewidth=3,label='Estimate of acceleration')
ax[2].set_xlabel("Discrete-time steps k",fontsize=14)
ax[2].set_ylabel("Acceleration",fontsize=14)
ax[2].tick_params(axis='both',labelsize=12)
#ax[0].set_yscale('log')
#ax[1].set_ylim(0,2)  
ax[2].grid()
ax[2].legend(fontsize=14)

fig.savefig('plots.png',dpi=600)


