import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Normalization
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import copy
import numpy as np
import pandas as pd

class EmbeddingGenerator:
    def __init__(self, epochs=5, batch_size=128, embedding_size=None):
        self.model = None
        self.categorical_columns = []
        self.numerical_columns = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_size = embedding_size  
        self.oe = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=9999, encoded_missing_value=9999, dtype=np.int32)
    
    def fit(self, X_train, y_train):
        self.X_train = X_train # keep a copy a training data
        self._get_encoded_categorical_data(X_train)
        self._get_numerical_data(X_train)
        self._create_model()
        self.model.fit(self._preprocess(X_train, case='train'), y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self
    
    def predict(self, X_test):
        predictions = self.model.predict(self._preprocess(X_test, case='predict'))
        return predictions

    def transform(self, X):
        X_copy = copy.deepcopy(X)
        self._get_encoded_categorical_data(X_copy)
        for c in self.categorical_columns:
            if c in X_copy.columns:
                embeddings = self._get_embedding_weights(self.model, c)
                embeddings = embeddings.add_prefix(f'{c}_')
                # for unseen categories during training
                extra_row = pd.DataFrame(np.zeros(embeddings.shape[1])).T
                extra_row.columns = embeddings.columns.tolist()
                extra_row.index = [9999]
                extra_row[f'{c}_index'] = 9999
                embeddings = pd.concat([embeddings, extra_row],axis=0)
            else:
                raise ValueError(f'{c} column is not in the input dataset')

            # TEST PRINT: print(f'embedding lookup table for {c}:\n{embeddings}')
            # mapping created based on training data
            cat_idx_map = {cat:idx for cat, idx in zip(sorted(set(self.X_train[c])), range(self.X_train[c].nunique()))}
            # for unseen categories during training
            cat_notin_X = [cat_tst for cat_tst in sorted(set(X_copy[c])) if cat_tst not in list(sorted(set(self.X_train[c])))]
            # TEST PRINT: print(f'Category in {c} seen in training but not present in test: {cat_notin_X}\n ')
            if len(cat_notin_X) != 0:
                print(f'Category in {c} seen in training but not present in test: {cat_notin_X}\n ')
                cat_idx_map = {cat:9999 for cat in cat_notin_X}
            cat_idx_map = pd.DataFrame(cat_idx_map.items(), columns=[c,f'{c}_index'])
            # TEST PRINT: print(f'final category to embeddings mapping:\n {cat_idx_map}')
            cat_embeddings_map = pd.merge(embeddings, cat_idx_map, on=f'{c}_index', how='inner').drop([f'{c}_index'],axis=1)
            # TEST PRINT: print(f'final category to embeddings mapping:\n {cat_embeddings_map}')
            X_copy = pd.merge(X_copy, cat_embeddings_map, on=c, how='inner')
        return X_copy

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
                                      output_dim=min(2, int(np.ceil((self.n_categories[idx])/2))), name=col)(input_cat)
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

    def _get_encoded_categorical_data(self, X, case='train'):
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.n_categories = [X[c].nunique() for c in self.categorical_columns]
        if case=='train':
            encoded_cat_data = self.oe.fit_transform(X[self.categorical_columns])
        if case=='predict':
            encoded_cat_data = self.oe.transform(X[self.categorical_columns])
        return [encoded_cat_data[:,i] for i in range(encoded_cat_data.shape[1])]
    
    def _get_numerical_data(self, X):
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        num_data = X[self.numerical_columns].values
        return [num_data]
        
    def _preprocess(self, X, case='train'):
        numeric_data = self._get_numerical_data(X)
        encoded_cat_data = self._get_encoded_categorical_data(X, case)
        return numeric_data + encoded_cat_data # total of 3 inputs (len(numeric_data + encoded_cat_data)=3) as the model architecture