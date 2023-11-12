from matplotlib import pyplot as plt

def main():
    eval_loss = [
        # 0.410,
        0.1,
        0.027,
        0.0265,
        0.0268,
        0.027,
        0.028,
        0.029,
        0.0307,
        0.030,
        0.0314,
    ]

    train_loss = [
        # 18.4127,
        # 2.0,
        0.1,
        0.0325,
        0.026,
        0.0223,
        0.02,
        0.0183,
        0.0172,
        0.0162,
        0.0157,
        0.0154,
    ]

    # Define the x-axis values
    epochs = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    # Plot eval_loss in blue and label it as "Eval Loss"
    plt.figure(figsize=(10,7))
    plt.plot(epochs, eval_loss, label='Eval Loss', color='blue')  # Plot with defined epochs

    # Plot train_loss in red and label it as "Train Loss"
    plt.plot(epochs, train_loss, label='Train Loss', color='red')  # Plot with defined epochs

    # Add labels and a legend
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)  # Set x-axis ticks
    plt.ylim(0, 0.05)  # Set y-axis range
    plt.legend()

    # Show the plot
    plt.savefig('plot.png')
    plt.show()

# Call the main function to execute the plotting
if __name__ == '__main__':
    main()
