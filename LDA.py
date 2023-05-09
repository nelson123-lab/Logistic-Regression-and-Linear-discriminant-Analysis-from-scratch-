class LDA:
  
  # Initializing the LDA objects
  def __init__(self):
     
     # Inializing the Unique classes as None
     self.U = None

     # Initializing the the class means as None
     self.c_m = None

     # Initializing the shared covariance matrix as None
     self.shared_cov = None

  def fit(self, data, target):
    # Finding the Uniques elements of the target.
    self.U = np.unique(target)

    # Initializing an empty array with dimensions (number of classes, number of features)
    self.c_m = np.zeros((len(self.U), data.shape[1]))
    
    # Finding the class means and class priors of the target.
    for idx, ele in enumerate(self.U): 

        # like_data selects the subset of the input data that belongs to that class.
        like_data = data[target == ele]

        # Means of each feature across all samples.
        self.c_m[ele] = like_data.mean(axis=0)
    
    # Intializing the covariance matrix with dimensions (number of features, number of features)
    self.shared_cov = np.zeros((data.shape[1], data.shape[1]))

    # Finding the class priors = c_p
    self.c_p = np.array([np.mean(target == ele) for ele in self.U])

    # get covariance MAtrix
    for idx, ele in enumerate(self.U):

        # like_data selects the subset of the input data that belongs to that class.
        like_data = data[target == ele]
        
        # Finding the Centered data matrix = C_d_m
        C_d_m = like_data - self.c_m[ele]

        # Covariance Matrix = C_M
        C_M = (C_d_m).T @ (C_d_m)

        # Updates shared covariance matrix of all classes.
        self.shared_cov +=  C_M

    # Finding average covariance matrix of all classes to get a better estimate.
    self.shared_cov /= data.shape[0] - len(self.U)

  # Defining the predict method.
  def predict(self, data):
      
      # Initializing an empty 2D array for storing Discriminant Scores.
      D_S = np.zeros((data.shape[0], len(self.U)))
      
      # Iterating over each sample
      for idx1, ele1 in enumerate(data):
          
          # Iterating over each class.
          for idx2, ele2 in enumerate(self.U):
              
              # Finding the difference between the input feature vector ele1 and the mean of class ele2.
              c_m = ele1 - self.c_m[ele2]
              
              # Finding the prior Probability = p_p of the class 
              p_p = np.log(self.c_p[idx2])

              # Finding the inverse of shared covariance matrix.
              shared_cov_inv = np.linalg.inv(self.shared_cov) 
              
              """
              Discriminant function D_k(x) = x.T @ w_k - 0.5 * w_k.T @ S_w @ w_k + ln(P(w_k))
              x - input feature vector
              w_k - weight vector or class means
              S_w - shared covariance matrix
              ln(p(w_k)) - prior probability
              """
              # Finding the discriminant function for each input feature and each class.
              D_S[idx1, idx2] = -0.5 * c_m @ shared_cov_inv @ c_m.T + p_p

        # Predict method returns the class with the maximum discriminant value for each input feature vectors.
      return self.U[np.argmax(D_S, axis=1)]
  
  # The Accuracy function finds the predictions from the X_test first and then compares them with the y_test.
  def Accuracy(self, X_test, y_test):
      # Finding the predictions.
      y_pred = self.predict(X_test)

      # Comparing matching values between the predictions and the y_test.
      c_pred = sum([1 for i in range(len(y_test)) if y_test[i] == y_pred[i]])

      # Returns the percetage of the Accurate predictions
      return (c_pred/len(y_test))*100