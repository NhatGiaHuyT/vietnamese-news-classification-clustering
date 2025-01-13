from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def train_cnn_model(X_train, y_train, max_num_words, max_sequence_length):
    model = Sequential([
        Embedding(input_dim=max_num_words, output_dim=128, input_length=max_sequence_length),
        Conv1D(128, 3, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.2),  

        Conv1D(256, 3, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.2), 

        Conv1D(512, 3, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.3), 

        GlobalAveragePooling1D(),
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.5),
        Dense(len(set(y_train)), activation='softmax')
    ])

    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_cnn_model.keras', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  

    model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=30,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    return model