import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
import datetime as dt
import pylab

kopf, arm, torus, bein = np.genfromtxt('puppe_masse.txt', unpack = True)
kopf *= .001 / 2.
arm *= .001 / 2.
torus *= .001 / 2.
bein *= .001 / 2.

kopf_ges = np.sum(kopf) / np.size(kopf)
kopf_err = np.sqrt(1. / (np.size(kopf) - 1.) * np.sum((kopf - kopf_ges)**2))
print("kopf_ges = " + str(kopf_ges) + "\nkopf_err = " + str(kopf_err))

arm_ges = np.sum(arm) / np.size(arm)
arm_err = np.sqrt(1. / (np.size(arm) - 1.) * np.sum((arm - arm_ges)**2))
print("arm_ges = " + str(arm_ges) + "\narm_err = " + str(arm_err))

torus_ges = np.sum(torus) / np.size(torus)
torus_err = np.sqrt(1. / (np.size(torus) - 1.) * np.sum((torus - torus_ges)**2))
print("torus_ges = " + str(torus_ges) + "\ntorus_err = " + str(torus_err))

bein_ges = np.sum(bein) / np.size(bein)
bein_err = np.sqrt(1. / (np.size(bein) - 1.) * np.sum((bein - bein_ges)**2))
print("bein_ges = " + str(bein_ges) + "\nbein_err = " + str(bein_err))