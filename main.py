import numpy as np
import matplotlib.pyplot as plt
import torch
import random

def train_data():
    v0 = random.uniform(5.0, 25.0)
    alpha = random.uniform(0.0, np.pi/2)
    g = 9.81
    t = random.uniform(0.0, 5.0)

    h = v0 * np.sin(alpha) * t - 0.5 * g * t**2

    return [v0,alpha,t], [h]

def generate_data():
    examples = [train_data() for _ in range(100)]

    x,y = zip(*examples)

    x = torch.tensor(x, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)

    return x,y

def main():

    x, y = generate_data()

    model = torch.nn.Sequential(
        torch.nn.Linear(3,16),
        torch.nn.ReLU(),
        torch.nn.Linear(16,1)
    )

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    number_of_epoches = 10000

    for epoch in range(number_of_epoches):
        optimizer.zero_grad()
        pred = model(x)
        current_loss = loss(pred, y)
        current_loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch} | Loss: {current_loss.item()}")

if __name__ == "__main__":
    main()