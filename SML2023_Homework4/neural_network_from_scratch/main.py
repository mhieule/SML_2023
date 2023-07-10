import argparse
from datetime import datetime
import numpy as np

from nn_model import NNModel
from nn_modules import LinearModule, SoftMaxModule, CrossEntropyLoss, TanhModule
from experiment import load_nn_model, save_nn_model, plot_loss, train, test


def run_training(module_specifications, train_data_file, train_label_file, number_epochs, batch_size, learning_rate):
    training_data = np.load(train_data_file)
    training_label = np.load(train_label_file)

    nn_model = NNModel(module_specifications)
    loss_module = CrossEntropyLoss()

    nn_model, loss_over_epochs = train(nn_model, loss_module, training_data, training_label, number_epochs, batch_size,
                                       learning_rate)

    plot_loss(loss_over_epochs)

    return nn_model


def run_evaluation(trained_model, test_data_file, test_label_file):
    test_data = np.load(test_data_file)
    test_label = np.load(test_label_file)

    test(trained_model, test_data, test_label)


if __name__ == '__main__':
    # General arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs='?', const=True, default=True, help="Run training")
    parser.add_argument("--test", nargs='?', const=True, default=True, help="Run model evaluation")
    parser.add_argument("--save", nargs='?', const=True, default=True, help="Save the trained model")
    parser.add_argument("--model_dir", type=str, default="./saved_models/", help="Directory to save trained model")
    parser.add_argument("--model_file", type=str, default=None, help="Pickle file to load specific model for testing")
    parser.add_argument("--train_data_file", type=str, default="./mnist_subset/train_images.npy")
    parser.add_argument("--train_label_file", type=str, default="./mnist_subset/train_labels.npy")
    parser.add_argument("--test_data_file", type=str, default="./mnist_subset/test_images.npy")
    parser.add_argument("--test_label_file", type=str, default="./mnist_subset/test_labels.npy")
    args = parser.parse_args()

    # Specification of neural network architecture
    # Simple neural network to start with
    module_specifications = [(LinearModule, {'number_input_neurons': 28 * 28, 'number_output_neurons': 10}),
                             (SoftMaxModule, {})]

    # Deeper neural network with tanh activation
    module_specifications = [(LinearModule, {'number_input_neurons': 28 * 28, 'number_output_neurons': 200}),
                             (TanhModule, {}),
                             (LinearModule, {'number_input_neurons': 200, 'number_output_neurons': 10}),
                             (SoftMaxModule, {})]


    # Hyperparameter for training
    number_epochs = 100
    batch_size = 100
    learning_rate = 0.2

    trained_model = None
    if args.train:
        trained_model = run_training(module_specifications, args.train_data_file, args.train_label_file,
                                     number_epochs, batch_size, learning_rate)

    if args.save and trained_model is not None:
        save_nn_model(trained_model, args.model_dir, datetime.now().strftime("%I:%M%p_%B_%d_%Y"))

    if args.test:
        if trained_model is None and args.model_file is not None:
            trained_model = load_nn_model(args.model_file)
        elif trained_model is None and args.model_file is None:
            raise Exception("No neural network has been trained and no model file has been provided for testing!")

        run_evaluation(trained_model, args.test_data_file, args.test_label_file)
