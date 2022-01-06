import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 10000,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())

    training_accuracies = event_acc.Scalars('training/gen_loss_total')
    validation_accuracies = event_acc.Scalars('training/mel_spec_error')
    loss_gen_s_accuracies = event_acc.Scalars("training/loss_gen_s")
    loss_gen_f_accuracies = event_acc.Scalars("training/loss_gen_f")
    loss_fm_s_accuracies = event_acc.Scalars("training/loss_fm_s")
    loss_fm_f_accuracies = event_acc.Scalars("training/loss_fm_f")
    f0_loss_accuracies = event_acc.Scalars("training/f0_loss")


    steps = len(training_accuracies)
    step_axis = np.arange(steps)*100
    gen_loss_total = np.zeros(steps)
    mel_spec_error = np.zeros(steps)
    loss_gen_s = np.zeros(steps)
    loss_gen_f = np.zeros(steps)
    loss_fm_f = np.zeros(steps)
    loss_fm_s = np.zeros(steps)
    f0_loss = np.zeros(steps)

    for i in range(steps):
        gen_loss_total[i] = training_accuracies[i][2] # value
        mel_spec_error[i] = validation_accuracies[i][2]
        loss_gen_s[i] = loss_gen_s_accuracies[i][2]
        loss_gen_f[i] = loss_gen_f_accuracies[i][2]
        loss_fm_s[i] = loss_fm_s_accuracies[i][2]
        loss_fm_f[i] = loss_fm_f_accuracies[i][2]
        f0_loss[i] = f0_loss_accuracies[i][2]

    plt.plot(step_axis, gen_loss_total[:], label='total loss')
    plt.plot(step_axis, mel_spec_error[:], label='mel spec loss')
    plt.plot(step_axis, loss_gen_s, label='adversarial loss MSD')
    plt.plot(step_axis, loss_gen_f, label='adversarial loss MPD')
    plt.plot(step_axis, loss_fm_s, label='feature matching loss MSD')
    plt.plot(step_axis, loss_fm_f, label='feature matching loss MPD')
    plt.stem(step_axis, f0_loss, label='pitch loss')

    plt.xlabel("Steps",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.ylim(bottom=0)
    plt.title("Training Progress",fontsize=20)
    plt.legend(loc='best', frameon=True)
    plt.grid(b=True)
    plt.show()


if __name__ == '__main__':
    # log_file = "./checkpoint_dir/logs/"
    if len(sys.argv) < 2:
        print("missing argument")
        exit(-1)
    plot_tensorflow_log(sys.argv[1])