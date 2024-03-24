from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dataset import CellDivisionDataset
import torch
import torch.optim as optim

def train_model(config, model):
    # Load the dataset
    dataset = CellDivisionDataset(
        config["data"]["annotations_path"],
        config["data"]["img_path"],
        transform=None,
        device=config["env"]["device"],
    )

    # Split dataset into training and validation sets
    train_size = int(config["train"]["size"] * len(dataset))
    val_size = len(dataset) - train_size

    # Create a generator on the required device
    generator = torch.Generator(device=config["env"]["device"])

    # Use this generator in the random_split function
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    # need to explicitly pass the generator to the dataloader for mps to work
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, generator=generator
    )

    # val_loader = DataLoader(
    #     val_dataset, batch_size=32, shuffle=False, generator=generator
    # )

    # Initialize the model
    model = model.to(config["env"]["device"])
    model.train()

    # Choose a loss function. For binary classification, BCELoss is commonly used.
    criterion = torch.nn.BCELoss()

    # Choose an optimizer (e.g., Adam) and link it to your model's parameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = config["train"]["epochs"]
    model = train_loop(num_epochs, model, criterion, optimizer, train_loader, config)




def train_loop(num_epochs, model, criterion, optimizer, train_loader, config):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Make sure inputs and labels are on the same device as the model
            inputs, labels = (
                inputs.to(config["env"]["device"]),
                labels.to(config["env"]["device"]),
            )

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(
                outputs.squeeze(), labels.float()
            )  # Use .squeeze() to remove any extra dims
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print("Finished Training")
    return model


# if __name__ == "__main__":
#     # This block runs if train.py is executed as a script
#     # Load configs and call train_model()
#     config = load_config("config.yaml")

#     train_model(config)
