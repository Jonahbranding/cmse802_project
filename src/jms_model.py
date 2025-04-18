#First model--simple implementaiton of JMS's model

#Begins by defining a class HandicapModel, with allele frequencies and selection parameters as attributes.
class HandicapModel:
    def __init__(self, p_A=0.5, p_B=0.5, p_C=0.5, s=0.1, t=0.2, u=0.3):
        """Initialize the model with allele frequencies and selection parameters."""
        self.p_A = p_A
        self.p_a = 1 - p_A
        self.p_B = p_B
        self.p_b = 1 - p_B
        self.p_C = p_C
        self.p_c = 1 - p_C
        
        self.s = s  # Selection coefficient for aa
        self.t = t  # Handicap coefficient for B males
        self.u = u  # Interaction coefficient for aaB males

# Defines a method for this class, fitness, according to the math above.
    def fitness(self, genotype):
        """Return the fitness value for a given genotype."""
        if genotype in ['AAB', 'AaB']:
            return (1, 1 - self.t)
        elif genotype in ['AAbb', 'Aabb']:
            return (1, 1 - self.s)
        elif genotype == 'aaB':
            return (1, (1 - self.s) * (1 - self.u * self.t))
        elif genotype == 'aabb':
            return (1 - self.s, 1 - self.s)
        else:
            raise ValueError("Invalid genotype")
    
# Defines another method for the class, mating_probabilities, which
    def mating_probabilities(self, female_genotype, male_genotypes):
        """Return mating probabilities based on female preference."""
        if female_genotype in ['CC', 'Cc']:
            return {g: 1 if 'B' in g else 0 for g in male_genotypes}
        else:
            return {g: 1 / len(male_genotypes) for g in male_genotypes}
    
# Defines a method "update frequencies" which updates over 1 generation
    def update_frequencies(self):
        """Update allele frequencies over one generation."""
        # Compute genotype frequencies before selection. This part uses the Hardy-Weinberg equation, which assumes random mating
        # (Of course, mating will be non-random with respect to genotype pretty soon, but this describes before selection)
        f_AA = self.p_A ** 2
        f_Aa = 2 * self.p_A * self.p_a
        f_aa = self.p_a ** 2
        f_BB = self.p_B ** 2
        f_Bb = 2 * self.p_B * self.p_b
        f_bb = self.p_b ** 2
        
        # Compute survival rates. This gives tuples for female fitness, male fitness.
        W_AAB, W_AaB = self.fitness("AAB")[1], self.fitness("AaB")[1]
        W_AAbb, W_Aabb = self.fitness("AAbb")[1], self.fitness("Aabb")[1]
        W_aaB, W_aabb = self.fitness("aaB")[1], self.fitness("aabb")[1]
        
        # Mean fitness of the population
        W_bar = (
            f_AA * self.p_B * W_AAB +
            f_Aa * self.p_B * W_AaB +
            f_AA * self.p_b * W_AAbb +
            f_Aa * self.p_b * W_Aabb +
            f_aa * self.p_B * W_aaB +
            f_aa * self.p_b * W_aabb
        )
        
        # Update allele frequencies
        self.p_A = (f_AA * self.p_B + f_Aa * self.p_B) / W_bar
        self.p_a = 1 - self.p_A
        self.p_B = (f_AA * self.p_B * W_AAB + f_Aa * self.p_B * W_AaB) / W_bar
        self.p_b = 1 - self.p_B
        
        def simulate(self, generations=100):
        """Run the model for a given number of generations and track allele changes."""
        history = []
        for _ in range(generations):
            self.update_frequencies()
            history.append((self.p_A, self.p_B, self.p_C))
        return np.array(history)

import matplotlib.pyplot as plt
import numpy as np

# Initialize the model with some starting parameters
model = HandicapModel(p_A=0.6, p_B=0.4, p_C=0.5, s=0.1, t=0.2, u=0.3)

# Run the simulation
generations = 200

#Assigns a name "history" to the output of the simulation over 200 generations
history = model.simulate(generations)

# Extracts allele frequencies; returns an array with one line per generation, and three columns corresponding to allele frequencies
p_A_values, p_B_values, p_C_values = history[:, 0], history[:, 1], history[:, 2]
