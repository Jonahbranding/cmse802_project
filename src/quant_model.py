#Here we extend the original model to incorporate additional realism, involving developmental constraints and quantitative sex-linked traits.
import numpy as np

#First, let's make a baseline model, that our other models will use. 
class BaseHandicapModel:
    def __init__(self, pop_size=1000, k=1.5, preference_strength=3.0, mutation_std=0.05, max_handicap=3.0):
        self.pop_size = pop_size
        self.k = k
        self.preference_strength = preference_strength
        self.mutation_std = mutation_std
        self.max_handicap = max_handicap
        self.history = []

#Provides survival probabilities as a function of k and h; it's exponential, because we want mortality to increase non-linearly. At extreme values, even slight increases in the handicap correspond to substantial increases in mortalitiy. 
    def survival_prob(self, h):
        return np.exp(-self.k * h)

#Similarly, probability of mating scales nonlinearly according to the trait value of the handicap.
    def mating_probabilities(self, handicaps):
        scaled = self.preference_strength * handicaps
        exp_vals = np.exp(scaled - np.max(scaled))
        return exp_vals / np.sum(exp_vals)
        
#Now, we move on to consider a sex-linked quantitative handicap
class SexLinkedHandicapModel(BaseHandicapModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.male_handicaps = np.full(self.pop_size, 0.1)  # Low initial value
        self.extinct = False  # Flag to track if the population has gone extinct

    def update_generation(self):
        # Calculate survival probabilities and apply them
        survival_probs = self.survival_prob(self.male_handicaps)
        survived = np.random.rand(self.pop_size) < survival_probs
        survivors = self.male_handicaps[survived]

        # If no survivors, set all handicaps to zero but continue the simulation
        if len(survivors) == 0:
            self.male_handicaps = np.zeros(self.pop_size)
            self.extinct = True  # Mark the population as extinct, but continue the simulation
            return  # No further calculations are needed for this generation
    
        # Calculate mating probabilities and select fathers
        mating_probs = self.mating_probabilities(survivors)
        fathers = np.random.choice(survivors, size=self.pop_size, p=mating_probs)

        # Generate offspring with mutations and clipping to max_handicap
        offspring = fathers + np.random.normal(0, self.mutation_std, size=self.pop_size)
        offspring = np.clip(offspring, 0, self.max_handicap)

        # Update male handicaps with the new offspring
        self.male_handicaps = offspring
        self.history.append(np.mean(self.male_handicaps))

# As a comparison, we'll look and see how a non-sex linked trait does; this will show if sex linkage can "make a difference"
class NonSexLinkedHandicapModel(BaseHandicapModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handicaps = np.full(self.pop_size, 0.1)
        self.sex = np.random.choice(['M', 'F'], size=self.pop_size)

    def update_generation(self):
        survival_probs = self.survival_prob(self.handicaps)
        survived_mask = np.random.rand(self.pop_size) < survival_probs
        self.handicaps = self.handicaps[survived_mask]
        self.sex = self.sex[survived_mask]

        males = self.handicaps[self.sex == 'M']
        females = self.handicaps[self.sex == 'F']

        if len(males) == 0 or len(females) == 0:
            self.handicaps = np.zeros(self.pop_size)
            return

        mating_probs = self.mating_probabilities(males)
        fathers = np.random.choice(males, size=self.pop_size, p=mating_probs)
        mothers = np.random.choice(females, size=self.pop_size)

        offspring_handicaps = (fathers + mothers) / 2 + np.random.normal(0, self.mutation_std, size=self.pop_size)
        offspring_handicaps = np.clip(offspring_handicaps, 0, self.max_handicap)
        offspring_sex = np.random.choice(['M', 'F'], size=self.pop_size)

        self.handicaps = offspring_handicaps
        self.sex = offspring_sex
        self.history.append(np.mean(self.handicaps))

# --- Run extended simulation ---
sex_linked = SexLinkedHandicapModel()
non_sex_linked = NonSexLinkedHandicapModel()

generations = 1000
for _ in range(generations):
    sex_linked.update_generation()
    non_sex_linked.update_generation()


#Print line to validate that the simulation has worked.
if __name__ == "__main__":
    sex_linked = SexLinkedHandicapModel()
    non_sex_linked = NonSexLinkedHandicapModel()

    generations = 1000
    for _ in range(generations):
        sex_linked.update_generation()
        non_sex_linked.update_generation()

    print("Simulation complete!")
    print("Final average handicap (sex-linked):", sex_linked.history[-1])
    print("Final average handicap (non-sex-linked):", non_sex_linked.history[-1])
