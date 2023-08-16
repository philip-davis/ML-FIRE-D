import numpy as np
import torch
# Assuming you have 'features.npz' file containing the features
data = np.load('0-epoch50_16.npz')

# Extract the batches of features
batches = data['arr_0']

# Initialize an empty array for concatenated features
concatenated_features = np.empty((batches.shape[0] * 16, 128)) # was 120,128

# Concatenate the features from each batch
for i, batch in enumerate(batches):
    start_idx = i * 16 # 120
    end_idx = start_idx + 16 # 120
    concatenated_features[start_idx:end_idx, :] = batch

# Reshape the concatenated features to the desired shape
final_feature_embedding = concatenated_features.reshape(16, -1) # was 120,-1
print(type(final_feature_embedding))
print(final_feature_embedding.shape)

embed = torch.tensor(final_feature_embedding)
torch.save(embed,'embed.pt')