import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python plot_graph.py <log_file_path>")
    sys.exit(1)

# Filepath of the log file
log_file_path = sys.argv[1]

def extract_losses(log_file_path):
    """Extract losses per epoch from the log file."""
    pretrain_losses = []
    comsys_losses = []
    final_accuracy = None

    with open(log_file_path, "r") as file:
        lines = file.readlines()

    # Flags to identify pretraining and fine-tuning sections
    pretrain_flag = True
    comsys_flag = False

    for line in lines:
        if "Epoch" in line and "Loss" in line:
            epoch_info = line.strip().split(", ")
            epoch = int(epoch_info[0].split("[")[1].split("/")[0])
            loss = float(epoch_info[-1].split(": ")[1])

            if "Pretraining completed" in line:
                pretrain_flag = False
                comsys_flag = True

            if pretrain_flag:
                if len(pretrain_losses) < epoch:
                    pretrain_losses.append(loss)

            if comsys_flag:
                if len(comsys_losses) < epoch:
                    comsys_losses.append(loss)

        if "Pretraining completed" in line:
            pretrain_flag = False
            comsys_flag = True

        if "Test Accuracy" in line:
            final_accuracy = float(line.split(": ")[1])

    return pretrain_losses, comsys_losses, final_accuracy

def plot_graphs(pretrain_losses, comsys_losses, final_accuracy):
    """Plot graphs for pretraining and fine-tuning losses."""
    # Plot pretraining losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(pretrain_losses) + 1), pretrain_losses, marker="o", label="Pretraining Loss")
    plt.title("Pretraining Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("pretraining_loss.png")

    # Plot fine-tuning losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(comsys_losses) + 1), comsys_losses, marker="o", label="Comsys Training Loss")
    plt.title("Comsys Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Annotate final accuracy
    plt.annotate(f"Final Accuracy: {final_accuracy:.2f}%", 
                 xy=(len(comsys_losses), comsys_losses[-1]), 
                 xytext=(len(comsys_losses) - 5, comsys_losses[-1] + 0.5),
                 arrowprops=dict(facecolor="black", arrowstyle="->"),
                 fontsize=10)

    plt.savefig("comsys_training_loss.png")

# Extract losses and accuracy
pretrain_losses, comsys_losses, final_accuracy = extract_losses(log_file_path)

# Plot the graphs
plot_graphs(pretrain_losses, comsys_losses, final_accuracy)