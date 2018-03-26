import numpy as np
import time
import csv

# Load CSV
print("Loading File...")

filename = 'data/mnist_train_0_1.csv'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=',')

outputs_train = data[:, 0]
inputs_train = np.delete(data, 0, 1)  # delete first column of data

filename = 'data/mnist_test_0_1.csv'
raw_data = open(filename, 'rt')
data = np.loadtxt(raw_data, delimiter=',')

outputs_test = data[:, 0]
inputs_test = np.delete(data, 0, 1)  # delete first column of data

print("Completed")


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):

    return sigmoid(x) * (1 - sigmoid(x))


def main():     # network

    # Initialize weights in network to small random numbers
    weights0 = np.random.uniform(low=-1.0, high=1.0, size=(784, 3))
    weights1 = np.random.uniform(low=-1.0, high=1.0, size=(3, 1))


    alpha = 0.005
    train_errors = np.array([])
    test_errors = np.array([])


    iter = 0


    while True:

        # for each training example named input_layer
        for i in range(len(inputs_train)):
            input_layer = inputs_train[i]
            y           = outputs_train[i]

            # FeedForward - prediction
            hidden_in = input_layer @ weights0
            hidden_layer = sigmoid(hidden_in)

            output_in = hidden_layer @ weights1
            output_layer = sigmoid(output_in)

            # Compute Error and Backpropogation - compute error
            output_error = y - output_layer
            train_errors = np.append(train_errors, output_error)


            # compute delta for all weights from hidden layer to output layer // backward pass
            output_delta = sigmoid_derivative(output_in) * output_error

            # compute delta for all weights from input layer to hidden layer // backward pass continued
            hidden_delta = (sigmoid_derivative(hidden_in) * weights1.T) * output_delta

            #update network weights // input layer not modified by error estimate
            weights1 += (alpha * hidden_layer).reshape((len(hidden_layer), 1)) * output_delta
            weights0 += (alpha * input_layer).reshape((len(input_layer), 1)) *  hidden_delta

            # DEBUG AREA
            # print("hidden_in", hidden_in)
            # print("hidden_layer", hidden_layer)
            # print("output_in", output_in)
            # print("output_layer", output_layer)
            # print("output_error", output_error)
            # print("output_delta", output_delta)
            # print("hidden_delta", hidden_delta)
            # print("1 shapes", weights1.T.shape, (alpha * hidden_layer).shape, output_delta.shape)
            # print("w1", weights1)
            # print("0 shapes", weights0.T.shape, (alpha * input_layer).shape, hidden_delta.shape)
            # print("w0", weights0)
            # print("w0 Shape:", weights0.shape)
            # print("w1 Shape:", weights1.shape)
            # DEBUG AREA


        print("Iteration:", iter)
        print("Training Mean Error:", np.mean(np.abs(train_errors)))

        correct_count = 0
        for i in range(len(inputs_test)):
            input_layer = inputs_test[i]
            y           = outputs_test[i]

            hidden_in = input_layer @ weights0
            hidden_layer = sigmoid(hidden_in)

            output_in = hidden_layer @ weights1
            output_layer = sigmoid(output_in)

            output_error = y - output_layer
            test_errors = np.append(test_errors, output_error)

            # Additional step activation to reduce error faster
            output_layer = np.piecewise(output_layer, [output_layer < 0.5, output_layer >= 0.5], [0, 1])

            if output_layer == y:
                # wrong_num += 1
                correct_count += 1

        # Compute error
        perc_correct = correct_count * 100 / len(inputs_test)
        print("Testing Mean Error:", np.mean(np.abs(test_errors)))
        print("Test % correct:", perc_correct)

        iter += 1
        if iter > 10 or perc_correct > 99.5:
            print("Stopping epochs..")
            break


    return

main()
