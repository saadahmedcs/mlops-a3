"""
Data training entrypoint wrapper.
Re-exports the training logic from the training module.
"""

from training.train import train

if __name__ == "__main__":
    train()
