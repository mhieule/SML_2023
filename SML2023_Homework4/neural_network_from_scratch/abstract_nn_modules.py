from abc import abstractmethod, ABC


class NNModule(ABC):
    @abstractmethod
    def fprop(self, input):
        """
        Propagate the input forward through the module

        :param input: Input of the module
        :return: Output after forward pass through module
        """
        raise NotImplementedError

    @abstractmethod
    def bprop(self, output_gradient):
        """
        Propagate the gradients from the output to the input of the module

        :param grad_out: Gradients at the output of the module
        :return: Gradient with respect to the inputs of the module
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_parameters(self):
        """
        Initialize the internal parameters of the module W, b

        :return: -
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_gradient_parameters(self, output_gradient):
        """
        Calculate the gradients with respect to the internal parameters W, b of the module
        Already accumulate the gradients over the batch

        :param grad_out: Gradients at the output of the module
        :return: Gradients with respect to the internal parameters accumulated over the batch
        """
        raise NotImplementedError

    @abstractmethod
    def apply_parameter_updates(self, batch_gradient_parameters, update_function):
        """
        Apply the update function to the internal parameters W, b of the module

        :param batch_grad_parameters: Gradients with respect to W, b accumulated over the batch
        :param update_function: The function to update the internal parameters
        :return: -
        """
        raise NotImplementedError


class NNModuleParameterFree(NNModule):
    # Activation functions do not have any internal parameters!
    def initialize_parameters(self):
        # Module does not have any internal parameters
        return

    def get_batch_gradient_parameters(self, output_gradient):
        # No gradients with respect to internal parameters
        return None

    def apply_parameter_updates(self, batch_gradient_parameters, update_function):
        # No internal parameters to update
        return


class LossModule(ABC):
    def set_target_values(self, targets):
        """
        Save the target values for later loss and loss gradient calculation

        :param targets: The target values
        :return:
        """
        self.targets = targets

    @abstractmethod
    def calculate_loss(self, input):
        """
        Calculate the loss given the input (output of neural network)

        :param input: Input of the module
        :return: Calculated loss
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_loss_gradient(self):
        """
        Calculate the gradient of the loss
        :return: Loss gradient
        """
        raise NotImplementedError
