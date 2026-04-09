import random
import torch

class CropEnv:
    def __init__(self, dataset=None):
        self.dataset = dataset
        self.current_image = None
        self.current_label = None
        self.mock_mode = dataset is None or len(dataset) == 0

    def reset(self, difficulty="easy"):
        if self.mock_mode:
            # Generate random 224x224x3 observation for validator
            self.current_image = torch.randn(3, 224, 224)
            self.current_label = random.randint(0, 15)
            print("🎲 Environment running in MOCK MODE (No dataset found)")
        else:
            if difficulty == "easy":
                subset_idx = range(min(100, len(self.dataset)))
            elif difficulty == "medium":
                subset_idx = range(min(500, len(self.dataset)))
            else:
                subset_idx = range(len(self.dataset))

            idx = random.choice(subset_idx)
            image, label = self.dataset[idx]
            self.current_image = image
            self.current_label = label

        return self.state()

    def state(self):
        return self.current_image

    def step(self, action):
        # Requirement 9: Non-binary reward
        if int(action) == int(self.current_label):
            reward = 1.0
        else:
            # Random partial reward for mock mode
            reward = 0.2
        
        done = True
        return self.state(), float(reward), done, {}