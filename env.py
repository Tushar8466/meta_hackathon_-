import random
import torch

class CropEnv:
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_image = None
        self.current_label = None

    # 🔁 Reset environment
    def reset(self, difficulty="easy"):
        if difficulty == "easy":
            subset_idx = range(min(100, len(self.dataset)))
        elif difficulty == "medium":
            subset_idx = range(min(500, len(self.dataset)))
        else:
            subset_idx = range(len(self.dataset))

        # Pick random index
        idx = random.choice(subset_idx)

        # Load image + label
        image, label = self.dataset[idx]

        # Store state
        self.current_image = image
        self.current_label = label

        return self.state()

    # 👁️ Return current observation
    def state(self):
        # In a real RL env, this would be the processed image tensor
        return self.current_image

    # ⚡ Take action and return reward
    def step(self, action):
        # OpenEnv Phase 1: Rewards must be between 0.0 and 1.0
        # Requirement 9: Must not be binary only (allow partial progress)

        current_class_name = self.dataset.classes[self.current_label]
        predicted_class_name = self.dataset.classes[int(action)]

        if int(action) == int(self.current_label):
            reward = 1.0
        elif current_class_name.split("_")[0] == predicted_class_name.split("_")[0]:
            # Partial reward: Correct crop genus, but wrong disease
            reward = 0.5
        else:
            # Base reward for effort/participation
            reward = 0.1

        done = True  # Single-step diagnosis tasks are usually done=True

        return self.state(), float(reward), done, {}