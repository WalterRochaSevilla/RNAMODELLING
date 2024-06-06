import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import keras
# Imprimir la versión de Keras
print(keras.__version__)

# Paso 1: Carga de datos
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Paso 2: Preprocesamiento de datos
# Convertir la columna de satisfacción en binaria en ambos conjuntos de datos
train_data['satisfaction'] = train_data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})
test_data['satisfaction'] = test_data['satisfaction'].map({'satisfied': 1, 'neutral or dissatisfied': 0})

# Manejo de valores nulos en 'Arrival Delay in Minutes'
train_data['Arrival Delay in Minutes'].fillna(train_data['Arrival Delay in Minutes'].mean(), inplace=True)
test_data['Arrival Delay in Minutes'].fillna(test_data['Arrival Delay in Minutes'].mean(), inplace=True)

# Seleccionar las características y la variable objetivo en el conjunto de entrenamiento
X_train = train_data.drop(columns=['satisfaction', 'id'])
y_train = train_data['satisfaction']

# Seleccionar las características y la variable objetivo en el conjunto de prueba
X_test = test_data.drop(columns=['satisfaction', 'id'])
y_test = test_data['satisfaction']

# Codificación de variables categóricas en ambos conjuntos de datos
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Asegurarse de que ambos conjuntos de datos tienen las mismas columnas después de la codificación
X_train, X_test = X_train.align(X_test, join='inner', axis=1)

# Normalización de las características numéricas en ambos conjuntos de datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Paso 3: Diseño de la red neuronal

model = keras.Sequential()
model.add(keras.layers.Dense(128, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization()) #normalización por paquetes
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Paso 4: Entrenamiento de la red con Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs=1000, batch_size=32,  validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])
#no usar validation split, directamente usar test
# Paso 5: Evaluación del modelo
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")

# Generar el reporte de clasificación
print(classification_report(y_test, y_pred, target_names=['neutral or dissatisfied', 'satisfied']))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Calcular métricas adicionales
tn, fp, fn, tp = conf_matrix.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'Specificity: {specificity}')
print(f'Precision: {precision}')
print(f'F1-Score: {f1_score}')