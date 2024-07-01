import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Normalization
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np

class EmbeddingGenerator:
    def __init__(self, embedding_size=None):
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.ordinal_encoders = {}
        self.embedding_outputs = {}
        self.embedding_size = embedding_size  
    
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
        # Inputs and embeddings for categorical data
        categorical_input = []
        embedding_layers = []
        for idx, col in enumerate(self.categorical_columns):
            input_cat = Input(shape=(1,), name=f'input_{col}')
            if self.embedding_size:
                embed_cat = Embedding(input_dim=len(self.label_encoders[col].classes_), output_dim=self.embedding_size)(input_cat)
            else: 
                embed_cat = Embedding(input_dim=len(self.label_encoders[col].classes_), 
                                      output_dim=np.min(50, np.int(np.ceil((self.n_categories[idx])/2))))(input_cat)
            flatten_cat = Flatten()(embed_cat)
            categorical_input.append(input_cat)
            embedding_layers.append(flatten_cat)
        
        # Inputs for numerical data
        numerical_input = Input(shape=(len(self.numerical_columns),), name='numerical_input')

        # Concatenate all inputs
        concatenated = Concatenate()([numerical_input] + embedding_layers)
        normalized = Normalization()(concatenated)
        
        # Hidden layers
        hidden1 = Dense(100, activation='relu')(normalized)
        hidden2 = Dense(100, activation='relu')(hidden1)
        
        # Output layer
        output = Dense(1, activation='sigmoid')(hidden2)
        
        # Create model
        model_inputs = [numerical_input] + categorical_input
        self.model = Model(inputs=model_inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        return self

    def get_categorical_data(self, X):
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.n_categories = [X[c].nunique() for c in self.categorical_columns]
        oe = OrdinalEncoder(categories='auto', dtype=np.int32)
        encoded_cat_data = oe.fit_transform(X[self.categorical_columns])
        return encoded_cat_data
    
    def get_numerical_data(self, X):
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        return X[self.numerical_columns].values, self.numerical_columns

    def preprocess(self, X):
        