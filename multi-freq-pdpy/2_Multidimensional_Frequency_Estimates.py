import matplotlib.pyplot as plt
import matplotlib
params = {'axes.titlesize':'14',
          'xtick.labelsize':'14',
          'ytick.labelsize':'14',
          'font.size':'14',
          'legend.fontsize':'medium',
          'lines.linewidth':'2',
          'font.weight':'normal',
          'lines.markersize':'10'
          }
matplotlib.rcParams.update(params)
plt.rcParams["mathtext.fontset"] = "cm"
plt.rc('font', family='serif')

import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


from multi_freq_ldpy.mdim_freq_est.RSpFD_solution import *
from multi_freq_ldpy.mdim_freq_est.SMP_solution import *
from multi_freq_ldpy.mdim_freq_est.SPL_solution import *

d = 3 # number of attributes
lst_k = [2, 5, 10] # number of values per attribute
input_data = [0, 3, 5] # real input values
eps = 1 # privacy guarantee

print('Real value:', input_data)
print('Sanitization w/ SPL solution and ADP protocol:', SPL_ADP_Client(input_data, lst_k, d, eps, optimal=True))
print('Sanitization w/ SMP solution and ADP protocol:', SMP_ADP_Client(input_data, lst_k, d, eps, optimal=True))
print('Sanitization w/ RSpFD solution and ADP protocol:', RSpFD_ADP_Client(input_data, lst_k, d, eps, optimal=True))



## Reading MS-FIMU dataset
df = pd.read_csv('datasets/db_ms_fimu.csv')

LE = LabelEncoder()
attributes = df.columns

for col in attributes:
    df[col] = LE.fit_transform(df[col])



# number of users
n = df.shape[0]
print('Number of Users =',n)

# number of attributes
d = len(attributes)
print('Number of Attributes =', d)

# domain size of attributes
lst_k = [len(df[att].unique()) for att in attributes]
print('Domain size of attributes =', lst_k)

print("\nPrivacy guarantees:")

# range of epsilon
lst_eps = np.arange(0.5, 5.1, 0.5)
print('Epsilon values =', lst_eps)

# Real normalized frequencies
real_freq = [np.unique(df[att], return_counts=True)[-1] / n for att in attributes]

# Repeat nb_seed times since DP protocols are randomized
nb_seed = 30

# Save Averaged Mean Squared Error (MSE_avg) between real and estimated frequencies per seed
dic_avg_mse = {seed:
                   {"SPL_GRR": [], "SPL_SUE": [], "SPL_OUE": [], "SPL_BLH": [], "SPL_OLH": [], "SPL_SS": [],
                    "SPL_ADP": [],
                    "SMP_GRR": [], "SMP_SUE": [], "SMP_OUE": [], "SMP_BLH": [], "SMP_OLH": [], "SMP_SS": [],
                    "SMP_ADP": [],
                    "RSpFD_GRR": [], "RSpFD_SUE_zero": [], "RSpFD_SUE_rnd": [], "RSpFD_OUE_zero": [],
                    "RSpFD_OUE_rnd": [], "RSpFD_ADP": []
                    }
               for seed in range(nb_seed)
               }

starttime = time.time()
for seed in range(nb_seed):
    print('Starting w/ seed:', seed)

    for eps in lst_eps:
        spl_reports = [SPL_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        spl_est_freq = SPL_ADP_Aggregator(spl_reports, lst_k, d, eps)
        dic_avg_mse[seed]["SPL_ADP"].append(
            np.mean([mean_squared_error(real_freq[att], spl_est_freq[att]) for att in range(d)]))

        smp_reports = [SMP_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        smp_est_freq = SMP_ADP_Aggregator(smp_reports, lst_k, d, eps)
        dic_avg_mse[seed]["SMP_ADP"].append(
            np.mean([mean_squared_error(real_freq[att], smp_est_freq[att]) for att in range(d)]))

        rspfd_reports = [RSpFD_ADP_Client(input_data, lst_k, d, eps) for input_data in df.values]
        rspfd_est_freq = RSpFD_ADP_Aggregator(rspfd_reports, lst_k, d, eps)
        dic_avg_mse[seed]["RSpFD_ADP"].append(
            np.mean([mean_squared_error(real_freq[att], rspfd_est_freq[att]) for att in range(d)]))

print('That took {} seconds'.format(time.time() - starttime))