'''
Visualisation tool for FOM against fiducial cuts.

Very simple, just for demonstration purposes.
'''

import numpy as np
import matplotlib.pyplot as plt

# radial cut values
rad_vals = [300, 400, 450, 500]
fom_vals = [1.9, 1.84, 1.88, 1.81]
effic_vals = [5.17, 5.14, 22.1, 44.54]

fig, ax1 = plt.subplots(figsize=(8, 5))

# radial
colour_left = 'tab:red'
ax1.set_xlabel('Radial cut values (mm)')
ax1.set_ylabel('Figure of Merit', color=colour_left)
ax1.plot(rad_vals, fom_vals, color=colour_left, linewidth=2)
ax1.tick_params(axis='y', labelcolor=colour_left)

# efficiency
ax2 = ax1.twinx()

colour_right = 'tab:blue'
ax2.set_ylabel('Efficiency (%)', color = colour_right)
ax2.plot(rad_vals, effic_vals, color = colour_right)
ax2.tick_params(axis='y', labelcolor = colour_right)

plt.tight_layout()
plt.show()
