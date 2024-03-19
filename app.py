from utils import rmse
from network import MLP

n_inputs = 5
layers = [4, 3, 2, 1]
epochs = 10
lr = 0.01

x_train = [[2.0, -3.0, 1.0, 1.0, 1],
           [1.0, -2.0, -2.0, 0.5, -2],
           [-1.0, -1.0, -4.0, -1.5, -1],
           [1.5, 1.0, 2.0, -3.5, 2],
        ]

y_train = [1.0, 0.0, 0.0, 1.0]

model = MLP(n_inputs, layers)


### Training

for epoch in range(1, epochs + 1):
    # Forward pass
    y_pred = [model(x)[0] for x in x_train]
    loss = rmse(y_train, y_pred)

    # Set zero grad
    for p in model.parameters():
        p.grad = 0.0

    # Backward pass
    loss.backward()

    # Gradient Descent and update parameter
    for p in model.parameters():
        p -= lr * p.grad
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss}, y_pred = {y_pred}")
