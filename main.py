from config import get_config
from train import train_val

if __name__ == "__main__":
    config = get_config()
    for bit in config["bit_list"]:
        print(config)
        train_val(config, bit)
