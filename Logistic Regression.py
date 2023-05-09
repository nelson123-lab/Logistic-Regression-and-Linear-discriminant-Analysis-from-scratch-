class LogisticRegression:
    def __init__(self, Lr = 0.1, epochs = 1000, L2_term = 0):
        # Lr : Learning rate
        # epochs : Number of epochs
        # L2_term : Regularization term

        self.Lr = Lr
        self.epochs = epochs
        self.L2_term = L2_term
        self.Weights = None
        self.bias = None
        self.targets = None
        
    # Stable Softmax function is used when more than two clases are defined.
    def Softmax(self, x):

        # Substracting max value of each row of z before exponentiating to prevent overflow.
        exponent = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exponent / np.sum(exponent, axis=1, keepdims=True)

    
    def fit(self, X, y, Lr = 0.01, epochs = 1000, L2_term = 0):
        self.Lr = Lr
        self.epochs = epochs
        self.L2_term = L2_term
        self.Loss = [] # To keep track of losses over epochs

        Dimensions = X.shape[1]
        # Finding the number of targets for classification
        self.targets = len(np.unique(y)) 

        # Initializing weights and biases
        self.Weights, self.bias = np.random.randn(Dimensions, self.targets), np.random.randn(self.targets)
        
        # One hot encoding the target values
        One_hot_y = self.hot_encoding(y)
        
        # No of epochs
        for _ in range(self.epochs):
            # Finding the matrix product output.
            Y = np.dot(X, self.Weights) + self.bias

            # Applying Softmax over the output
            y_pred = self.Softmax(Y)

            # Finding the cross entropy loss
            loss = self.Cross_entropy_loss(y_pred, One_hot_y)

            # Finding the gradients dw and db for updating weights.
            derivatives = self.Derivatives(X, y_pred, One_hot_y)

            # Updating the weights using the gradient - dw
            self.Weights -= self.Lr * (derivatives["dW"] + self.L2_term * self.Weights)

            # Updating the bias using the gradient - db
            self.bias -= self.Lr * derivatives["db"]

            # Apending the loss per 100 iterations
            if _ % 100 == 0:
                self.Loss.append(loss)
    
    # Function fro one hot encoding the data.
    def hot_encoding(self, y):
        return np.eye(self.targets)[y]
    
    # Function for finding the cross entropy loss
    def Cross_entropy_loss(self, y_pred, One_hot_y):
        # Cross entropy loss equation
        loss = -np.sum(One_hot_y * np.log(y_pred)) / y_pred.shape[0]

        # Regularization Term
        L2_term = (self.L2_term / 2) * np.sum(np.square(self.Weights))
        return loss + L2_term

    
    def Derivatives(self, X, y_pred, One_hot_y):
        # Finding the derivative of weights
        dw = np.dot(X.T, (y_pred - One_hot_y) / y_pred.shape[0])

        # Finding the derivative of bias
        db = np.sum((y_pred - One_hot_y) / y_pred.shape[0], axis=0)
        return {"dW": dw, "db": db}

    # The prediction function finds the predcitions by appplying armax function over the softmax output.
    def predict(self, X):
        return np.argmax(self.Softmax(np.dot(X, self.Weights) + self.bias), axis=1)
    
    # The Accuracy function finds the predictions from the X_test first and then compares them with the y_test.
    def Accuracy(self, X_test, y_test):
        # Finding the predictions.
        y_pred = self.predict(X_test)

        # Comparing matching values between the predictions and the y_test.
        c_pred = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred[i]])

        # Returns the percetage of the Accurate predictions
        return (c_pred/len(y_test))*100