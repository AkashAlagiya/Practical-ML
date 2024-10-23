import pandas as pd
import math


class NaiveBayes:
    """
    Naive Bayes classifier for continuous and discrete features using pandas
    """

    def __init__(self, continuous=None):
        """
        :param continuous: list containing a bool for each feature column to be analyzed. True if the feature column
                           contains a continuous feature, False if discrete
        """
        self.class_priors = {}
        self.feature_types = []
        self.feature_likelihoods = {}
        self.classes = []
        self.continuous = continuous

        pass

    def fit(self, data: pd.DataFrame, target_name: str):
        """
        Fitting the training data by saving all relevant conditional probabilities for discrete values or for continuous
        features.
        :param data: pd.DataFrame containing training data (including the label column)
        :param target_name: str Name of the label column in data
        """
        # Separate features and target
        X = data.drop(columns=[target_name])
        y = data[target_name]
        
        # Store class labels
        self.classes = y.unique()
        
        # Calculate Prior Probability of Classes P(y)
        total_size = len(y)
        self.class_priors = {cls: sum(y == cls) / total_size for cls in self.classes}
        
        # Determine features' types (continuous or discrete)
        if self.continuous is None:
            self.feature_types = ['continuous' if pd.api.types.is_float_dtype(X[col]) else 'discrete' for col in X.columns]
        else:
            self.feature_types = ['continuous' if is_cont else 'discrete' for is_cont in self.continuous]
        
        # Initialize feature likelihoods dictionary
        self.feature_likelihoods = {cls: {} for cls in self.classes}
        
        # Calculate likelihoods for each feature
        for cls in self.classes:
            # Subset the data by class
            X_class = X[y == cls]
            
            for i, col in enumerate(X.columns):
                if self.feature_types[i] == 'discrete':
                    # Calculate frequency counts for discrete features
                    counts = X_class[col].value_counts()
                    total_count = len(X_class)
                    self.feature_likelihoods[cls][col] = {k: v / total_count for k, v in counts.items()}
                else:
                    # Calculate mean and standard deviation for continuous features
                    mean = X_class[col].mean()
                    std = X_class[col].std()
                    self.feature_likelihoods[cls][col] = (mean, std)
        pass

    def predict_probability(self, data: pd.DataFrame):
        """
        Calculates the Naive Bayes prediction for a whole pd.DataFrame.
        :param data: pd.DataFrame to be predicted
        :return: pd.DataFrame containing probabilities for all categories as well as the classification result
        """
        results = []
        
        for _, row in data.iterrows():
            # Calculate the posterior probability for each class
            posteriors = {}
            
            for cls in self.classes:
                # Start with the prior probability of the class
                posterior = math.log(self.class_priors[cls])  # Use log to avoid underflow
                
                # Calculate the likelihood for each feature
                for i, col in enumerate(data.columns):
                    if self.feature_types[i] == 'discrete':
                        # Handle discrete features
                        value = row[col]
                        likelihood = self.feature_likelihoods[cls][col].get(value, 1e-6)  # Small value for unseen data
                    else:
                        # Handle continuous features using Gaussian distribution
                        mean, std = self.feature_likelihoods[cls][col]
                        x = row[col]
                        if std > 0:  # Avoid division by zero
                            likelihood = (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) * 2) / (2 * std * 2))
                        else:
                            likelihood = 1e-6
                    posterior += math.log(likelihood)
                
                # Store the posterior probability for the class
                posteriors[cls] = math.exp(posterior)
            
            # Normalize to get the probabilities
            total = sum(posteriors.values())
            posteriors = {cls: prob / total for cls, prob in posteriors.items()}
            
            # Find the class with the highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            posteriors["predicted_class"] = predicted_class
            
            results.append(posteriors)
        
        return pd.DataFrame(results)
        pass

    def evaluate_on_data(self, data: pd.DataFrame, test_labels):
        """
        Predicts a test DataFrame and compares it to the given test_labels.
        :param data: pd.DataFrame containing the test data
        :param test_labels:
        :return: tuple of overall accuracy and confusion matrix values
        """
         # Predict probabilities and get predicted classes
        predictions = self.predict_probability(data)
        predicted_classes = predictions["predicted_class"]
        
        predicted_classes = predicted_classes.reset_index(drop=True)
        test_labels = test_labels.reset_index(drop=True)
        
        # Calculate accuracy
        accuracy = (predicted_classes == test_labels).mean()
        
        # Generate confusion matrix
        confusion_matrix = pd.crosstab(predicted_classes, test_labels, rownames=['Predicted'], colnames=['Actual'])
        
        return accuracy, confusion_matrix
        pass
