# file leak.py
import torch

def run():
    # input_size * hidden_size >= 2048 cause leak
    # also observed extreme performace degeneration
    input_size = 16
    hidden_size = 128
    gru = torch.nn.GRU(input_size, hidden_size)
    inputs = torch.randn(1, 1, input_size)

    counter = 0
    while True:
        counter += 1
        _, out = gru(inputs)
        if (counter % 100) == 0:
            print(f"{counter} shape: {out.shape}")


if __name__ == "__main__":
    with torch.no_grad():
        run()
