# This module will test the model

# Author: Jonah Branding
# Date: March 2025

#This section was generated with assistance from ChatGPT (version 3.5, on March 23rd, 2025).

import unittest
import numpy as np
from model import HandicapModel

class TestHandicapModel(unittest.TestCase):

#Fitness values
    def setUp(self):
        """Set up a HandicapModel instance before each test."""
        self.model = HandicapModel(s=0.1, t=0.2, u=0.3)

    def test_fitness_values(self):
        """Test if fitness function returns correct values."""
        self.assertEqual(self.model.fitness("AAB"), (1, 0.8))  # 1 - t
        self.assertEqual(self.model.fitness("aaB"), (1, (1 - 0.1) * (1 - 0.3 * 0.2)))
        self.assertEqual(self.model.fitness("aabb"), (0.9, 0.9))  # 1 - s

    def test_invalid_fitness_input(self):
        """Ensure invalid genotypes raise an error."""
        with self.assertRaises(ValueError):
            self.model.fitness("XYZ")  # Invalid input

# Mating probabilities
    def test_mating_probabilities(self):
        """Test if CC and Cc females select only B males."""
        male_genotypes = ["AAB", "AAbb", "AaB", "aaB", "aabb"]
        
        # CC females must only select B males
        expected_selection = {"AAB": 1, "AaB": 1, "aaB": 1, "AAbb": 0, "aabb": 0}
        self.assertEqual(self.model.mating_probabilities("CC", male_genotypes), expected_selection)
        
        # cc females must select in proportion
        random_selection = self.model.mating_probabilities("cc", male_genotypes)
        self.assertEqual(sum(random_selection.values()), 1.0)  # Probabilities must sum to 1
