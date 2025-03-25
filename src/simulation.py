# This module runs the simulation per the specificaitons in 'model.py'

# Author: Jonah Branding
# Date: March 2025

from model import HandicapModel
import matplotlib.pyplot as plt

# Initialize the model with some starting parameters
model = HandicapModel(p_A=0.6, p_B=0.4, p_C=0.5, s=0.1, t=0.2, u=0.3)

# Run the simulation
generations = 200

#Assigns a name "history" to the output of the simulation over 200 generations
history = model.simulate(generations)

# Extracts allele frequencies; returns an array with one line per generation, and three columns corresponding to allele frequencies
p_A_values, p_B_values, p_C_values = history[:, 0], history[:, 1], history[:, 2]
