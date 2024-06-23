import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from sklearn.preprocessing import LabelEncoder
import numpy as np

class EmbeddingGenerator:
    def __init__(self):
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.label_encoders = {}
        self.embedding_outputs = {}
        self.embedding_size = 10  # This can be adjusted based on the specific use case
    
    def fit(self, X_train, y_train):
        self.get_categorical_data(X_train)
        self.get_numerical_data(X_train)
        self.create_model()
        
        # Convert categorical variables to embeddings
        X_train_embeddings = self._transform_categorical(X_train)
        X_train_numerical = X_train[self.numerical_columns].values
        
        # Concatenate numerical and embedding features
        X_train_combined = np.concatenate([X_train_embeddings, X_train_numerical], axis=1)
        
        self.model.fit(X_train_combined, y_train, epochs=10, batch_size=32, verbose=1)
        return self
    
    def transform(self, X):
        X_embeddings = self._transform_categorical(X)
        X_numerical = X[self.numerical_columns].values
        X_combined = np.concatenate([X_embeddings, X_numerical], axis=1)
        return X_combined
    
    def predict(self, X_test):
        X_test_transformed = self.transform(X_test)
        predictions = self.model.predict(X_test_transformed)
        return predictions
    
    def create_model(self):
        # Inputs for numerical data
        numerical_input = Input(shape=(len(self.numerical_columns),), name='numerical_input')
        
        # Inputs and embeddings for categorical data
        embedding_layers = []
        for col in self.categorical_columns:
            input_cat = Input(shape=(1,), name=f'input_{col}')
            embed_cat = Embedding(input_dim=len(self.label_encoders[col].classes_), output_dim=self.embedding_size)(input_cat)
            flatten_cat = Flatten()(embed_cat)
            embedding_layers.append(flatten_cat)
            self.embedding_outputs[col] = flatten_cat

        # Concatenate all inputs
        concatenated = Concatenate()([numerical_input] + embedding_layers)
        
        # Hidden layers
        hidden1 = Dense(100, activation='relu')(concatenated)
        hidden2 = Dense(100, activation='relu')(hidden1)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(hidden2)
        
        # Create model
        model_inputs = [numerical_input] + [self.embedding_outputs[col].input for col in self.categorical_columns]
        self.model = Model(inputs=model_inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return self

    def get_categorical_data(self, X):
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in self.categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = X[col].values
        return X[self.categorical_columns].values, self.categorical_columns
    
    def get_numerical_data(self, X):
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return X[self.numerical_columns].values, self.numerical_columns

    def _transform_categorical(self, X):
        X_transformed = []
        for col in self.categorical_columns:
            X_transformed.append(self.label_encoders[col].transform(X[col].values).reshape(-1, 1))
    return np.hstack(X_transformed)