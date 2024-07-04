import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Normalization
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np
import pandas as pd

class EmbeddingGenerator:
    def __init__(self, embedding_size=None):
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.embedding_size = embedding_size  
    
    def fit(self, X_train, y_train):
        self.X_train = X_train # keep a copy a training data
        self._get_encoded_categorical_data(X_train)
        self._get_numerical_data(X_train)
        self._create_model()
        self.model.fit(self._preprocess(X_train), y_train, epochs=10, batch_size=32, verbose=1)
        return self
    
    def predict(self, X_test):
        predictions = self.model.predict(self._preprocess(X_test))
        return predictions

    def transform(self, X):
        X_copy = X.copy()
        self._get_encoded_categorical_data(X_copy)
        for c in self.categorical_columns:
            if c in X_copy.columns:
                embeddings = self._get_embedding_weights(self.model, c)
                embeddings = embeddings.add_prefix(f'{c}_')
            else:
                raise ValueError(f'{c} column is not in the input dataset')
            # mapping created based on training data
            cat_idx_map = {cat:idx for cat, idx in zip(sorted(set(self.X_train[c])), range(self.X_train[c].nunique()))}
            cat_idx_map = pd.DataFrame(cat_idx_map.items(), columns=[c,f'{c}_index'])
            cat_embeddings_map = pd.merge(embeddings, cat_idx_map, on=f'{c}_index', how='left').drop([f'{c}_index'],axis=1)
            X_copy = pd.merge(X_copy, cat_embeddings_map, on=c, how='inner')
        return X_copy

    # Update: Need something better than mapping categories and their numerical indices using dictionary. Its too slow!
    def _get_embedding_weights(self, model, embed_layer_name):
        X_cat_emb = pd.DataFrame(model.get_layer(embed_layer_name).get_weights()[0]).reset_index()
        return X_cat_emb        
    
    def _create_model(self):
        # Inputs and embeddings for categorical data
        categorical_input = []
        embedding_layers = []
        for idx, col in enumerate(self.categorical_columns):
            input_cat = Input(shape=(1,), name=f'input_{col}')
            if self.embedding_size is not None:
                embed_cat = Embedding(input_dim=self.n_categories[idx], output_dim=self.embedding_size, name=col)(input_cat)
            else: 
                embed_cat = Embedding(input_dim=self.n_categories[idx], 
                                      output_dim=min(50, int(np.ceil((self.n_categories[idx])/2))), name=col)(input_cat)
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
        output = Dense(1, activation='linear')(hidden2)

        # Create model
        model_inputs = [numerical_input] + categorical_input
        self.model = Model(inputs=model_inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])      
        return self

    def _get_encoded_categorical_data(self, X):
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.n_categories = [X[c].nunique() for c in self.categorical_columns]
        oe = OrdinalEncoder(categories='auto', dtype=np.int32)
        encoded_cat_data = oe.fit_transform(X[self.categorical_columns])
        return [encoded_cat_data[:,i] for i in range(encoded_cat_data.shape[1])]
    
    def _get_numerical_data(self, X):
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        num_data = X[self.numerical_columns].values
        return [num_data]
        
    def _preprocess(self, X):
        numeric_data = self._get_numerical_data(X)
        encoded_cat_data = self._get_encoded_categorical_data(X)
        print("total inputs: ", )
        return numeric_data + encoded_cat_data # total of 3 inputs (len(numeric_data + encoded_cat_data)=3) as the model architecture