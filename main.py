import os
import sys
from models.lstm.train import run_training


if __name__ == "__main__" :
    args = sys.argv[1:]
    run_training(args)
