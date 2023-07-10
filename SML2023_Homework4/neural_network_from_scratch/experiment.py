import pickle
from functools import partial
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def load_nn_model(model_file):
    with open(model_file, 'rb') as f:
        trained_model = pickle.load(f)

    return trained_model


def save_nn_model(trained_model, model_directory, file_name):
    file_path = model_directory + '/' + file_name
    with open(file_path, 'wb') as f:
        pickle.dump(trained_model, f)


def plot_loss(loss_over_epochs):
    epochs = range(1, len(loss_over_epochs) + 1)
    plt.plot(epochs, loss_over_epochs)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()


def train(nn_model, loss_module, training_data, training_label, number_epochs, batch_size, learning_rate):
    nn_model.initialize_neural_network()

    def gradient_descent(parameter, parameter_gradient, learning_rate):
        # standard gradient descent
        return parameter - learning_rate * parameter_gradient

    update_function = partial(gradient_descent, learning_rate=learning_rate)
    number_training_data = training_data.shape[0]
    loss_over_epochs = []

    # Run training epochs
    for epoch in tqdm(range(0, number_epochs), desc="Training Progress"):
        # Use random permutation of the data
        indices = np.random.permutation(np.arange(number_training_data))
        epoch_loss = 0

        # Iterate over all mini-batches from this epoch
        number_mini_batches = int(np.ceil(float(number_training_data) / batch_size))
        for batch_number in range(0, number_mini_batches):
            # Fetch data (randomized)
            batch_indices = indices[np.arange(batch_number * batch_size, min((batch_number + 1) * batch_size, number_training_data))]
            batch_data = training_data[batch_indices, :]
            batch_label = training_label[batch_indices]

            # Forward propagation
            model_output = nn_model.fprop(batch_data)

            # Compute loss
            loss_module.set_target_values(batch_label)
            batch_loss = loss_module.calculate_loss(model_output)
            epoch_loss += np.mean(batch_loss)

            # Calculate loss gradient to backpropagate through network
            loss_gradient = loss_module.calculate_loss_gradient()
            nn_model.bprop(loss_gradient)

            # Update model parameters implicitly using back-propagated gradients
            nn_model.update_internal_parameters(update_function)

        epoch_loss /= int(np.ceil(float(number_training_data) / batch_size))
        loss_over_epochs.append(epoch_loss)

    return nn_model, loss_over_epochs


def test(trained_model, test_data, test_label):
    batch_size = 100
    number_test_data = test_data.shape[0]

    correct_predictions = 0
    number_mini_batches = int(np.ceil(float(number_test_data) / batch_size))
    for batch_number in tqdm(range(0, number_mini_batches), desc="Evaluation Progress"):
        batch_indices = np.arange(batch_number * batch_size, min((batch_number + 1) * batch_size, number_test_data))

        batch_data = test_data[batch_indices, :]
        batch_label = test_label[batch_indices]

        model_out = trained_model.fprop(batch_data)
        predictions = np.argmax(model_out, 1)

        target_indices = np.argmax(batch_label, axis=1)
        equality_check = np.equal(predictions, target_indices).astype(int)
        correct_predictions += np.sum(equality_check)

    print("{} prediction errors within {} evaluation data points".format((number_test_data - correct_predictions),
                                                                         number_test_data, ))
    print("Test accuracy %f" % (float(correct_predictions) / number_test_data))
