from matplotlib import pyplot as plt

def main():
    eval_loss = [0.01106424629688263, 
    0.002036006422713399,
    0.0011896648211404681,
    0.000784509174991399,
    0.0004359440063126385,
    0.00046205782564356923,
    0.00036975322291254997,
    0.00039013446075841784,
    0.00040770869236439466,
    0.0004132803878746927,
    0.00039774063043296337,
    0.0003999907639808953
    ]

    train_loss = [
        0.0198,
        0.0052,
        0.0039,
        0.0024,
        0.0009,
        0.0008,
        0.0005,
        0.0003,
        0.0003,
        0.0004,
        0.0002,
        0.0003
    ]

    # Plot eval_loss in blue and label it as "Eval Loss"
    plt.figure(figsize=(10,7))
    plt.plot(eval_loss, label='Eval Loss', color='blue')

    # Plot train_loss in red and label it as "Train Loss"
    plt.plot(train_loss, label='Train Loss', color='red')

    # Add labels and a legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()
    print()

if __name__ == '__main__':
    main()