import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # 导入评估指标
import shap
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')

file_path = "C:\\Users\\192052\\Desktop\\Explainable Course\\CS2_35.csv"
df = pd.read_csv(file_path)

input_features = ['CCCT', 'CVCT', 'R']
output_feature = 'capacity'

for col in input_features + [output_feature]:
    if col not in df.columns:
        raise ValueError(f" '{col}' not found。")

X_raw = df[input_features].values
y_raw = df[output_feature].values.reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(df['CCCT'], df['capacity'], alpha=0.6)
plt.title('Scatter Plot: CCCT vs Capacity (Original Data)')
plt.xlabel('CCCT')
plt.ylabel('Capacity')
plt.grid(True)
plt.show()

correlation = df['CCCT'].corr(df['capacity'])

plt.figure(figsize=(8, 6))
sns.heatmap(df[['CCCT', 'CVCT', 'R', 'capacity']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features and Target')
plt.show()

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X_raw)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_raw)

def create_sequences(input_data, output_data, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(input_data) - seq_length + 1):
        X_seq.append(input_data[i:i + seq_length])
        y_seq.append(output_data[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 1
X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)

X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

train_size = int(len(X_tensor) * 0.8)
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = len(input_features)
hidden_size = 100
num_layers = 3
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
batch_size = 16

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_losses = []

print("\nstart...")
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("finish")

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'MSE: {test_loss.item():.4f}')

    test_outputs_unscaled = scaler_y.inverse_transform(test_outputs.numpy())
    y_test_unscaled = scaler_y.inverse_transform(y_test.numpy())

    rmse = np.sqrt(mean_squared_error(y_test_unscaled, test_outputs_unscaled))
    mae = mean_absolute_error(y_test_unscaled, test_outputs_unscaled)
    r2 = r2_score(y_test_unscaled, test_outputs_unscaled)

    print(f'=RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R-squared: {r2:.4f}')

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label='real (Capacity)')
    plt.plot(test_outputs_unscaled, label='predicted (Capacity)')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity')
    plt.legend()
    plt.grid(True)
    plt.show()

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)].to(X_train.device)
explainer = shap.DeepExplainer(model, background)

shap_values_test = explainer.shap_values(X_test, check_additivity=False)
shap_values_test_reshaped = shap_values_test.reshape(X_test.shape[0], -1)

shap_values_train = explainer.shap_values(X_train, check_additivity=False)
shap_values_train_reshaped = shap_values_train.reshape(X_train.shape[0], -1)
print(" SHAP finish。")

feature_names = input_features

shap.summary_plot(shap_values_test_reshaped, features=X_test.reshape(X_test.shape[0], -1).numpy(), feature_names=feature_names)

shap.initjs()
expected_value_item = explainer.expected_value.item() if isinstance(explainer.expected_value,
                                                                    torch.Tensor) else explainer.expected_value
shap.force_plot(expected_value_item, shap_values_test_reshaped[0, :], X_test.reshape(X_test.shape[0], -1).numpy()[0, :],
                feature_names=feature_names)

shap.dependence_plot(
    "CCCT",
    shap_values_train_reshaped,
    X_train.reshape(X_train.shape[0], -1).numpy(),
    feature_names=feature_names,
)
plt.show()

shap.dependence_plot(
    "R",
    shap_values_train_reshaped,
    X_train.reshape(X_train.shape[0], -1).numpy(),
    feature_names=feature_names,
)
plt.show()

shap.dependence_plot(
    "CVCT",
    shap_values_train_reshaped,
    X_train.reshape(X_train.shape[0], -1).numpy(),
    feature_names=feature_names,
)
plt.show()

excel_output_path = "C:\\Users\\192052\\Desktop\\Explainable Course\\LSTM_Model_Results.xlsx"

with pd.ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
    df_train_losses = pd.DataFrame({
        'Epoch': range(1, num_epochs + 1),
        'Training Loss (MSE)': train_losses
    })
    df_train_losses.to_excel(writer, sheet_name='Training Loss', index=False)

    df_metrics = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'MAE', 'R-squared'],
        'Value': [test_loss.item(), rmse, mae, r2]
    })
    df_metrics.to_excel(writer, sheet_name='Evaluation Metrics', index=False)

    df_predictions = pd.DataFrame({
        'Sample Index': range(len(y_test_unscaled)),
        'Actual Capacity': y_test_unscaled.flatten(),
        'Predicted Capacity': test_outputs_unscaled.flatten()
    })
    df_predictions.to_excel(writer, sheet_name='Predictions_TestSet', index=False)

    df_shap_summary = pd.DataFrame(shap_values_test_reshaped, columns=[f'SHAP_{f}' for f in feature_names])
    df_features_test = pd.DataFrame(X_test.reshape(X_test.shape[0], -1).numpy(), columns=[f'Feature_{f}' for f in feature_names])

    df_shap_data = pd.concat([df_features_test, df_shap_summary], axis=1)
    df_shap_data.to_excel(writer, sheet_name='SHAP_Summary_Data_TestSet', index=False)

    df_shap_train_features = pd.DataFrame(X_train.reshape(X_train.shape[0], -1).numpy(), columns=[f'Feature_{f}' for f in feature_names])
    df_shap_train_values = pd.DataFrame(shap_values_train_reshaped, columns=[f'SHAP_{f}' for f in feature_names])

    df_shap_dependence_data = pd.concat([df_shap_train_features, df_shap_train_values], axis=1)
    df_shap_dependence_data.to_excel(writer, sheet_name='SHAP_Dependence_Data_TrainSet', index=False)


print(f"\nFile output to: {excel_output_path}")