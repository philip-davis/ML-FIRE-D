# ML-FIRE-D
The experimental setup details of our models are as follow
(i) ResNet18: pre-trained ResNet18 is obtained from https://pytorch.org/vision/stable/models.html;
(ii) ResNet50: pre-trained ResNet50 is obtained from https://pytorch.org/vision/stable/models.html;
(iii) SimCLR: learning rate: 0.05; hidden dimension: 128; number of layers: 1; augmentation: random
horizontal flip and resized crop; batch size: 32; weight decay: 1e-4; dropout rate: 0.5; contrastive objective:
SimCLR and N-pair losses; perturbation ratio: 0.1; using cosine annealing: yes; epoch = 50;
(iv) MVGRL: learning rate:0.01; hidden layers: 2; hidden dimension: 32; epoch: 1000 augmentation: identity
and personalized PageRank (PPR); batch size: 5; contrastive objective: Jensen-Shannon Divergence
(JSD); epoch = 1000;
(v) GraphCL: learning rate: 0.05; hidden dimension: 128; number of layers: 1; augmentation: node dropping
and edge perturbation; batch size: 32; dropout rate: 0.5; contrastive objective: Jensen-Shannon Divergence
(JSD); perturbation ratio: 0.1; epoch = 1000;
(vi) BRGL: learning rate:0.01; hidden layers: 1; 2; hidden dimension: 256; epoch: 1000 augmentation: edge
removing: 0.5 and feature masking: 0.1; batch size: 5; dropout rate: 0.2; contrastive objective: Jensen-
Shannon Divergence (JSD); epoch = 1000;
(vii) InfoGraph: learning rate:0.01; hidden layers: 2; hidden dimension: 32 epoch: 1000; batch size: 5;
contrastive objective: Jensen-Shannon Divergence (JSD); epoch = 1000;
(viii) Autoencoder: learning rate: (0.1.0.01,0.00,0.0001,0.00001); encoder hidden layers 3; decoder hidden
layers: 3; batch size: 5; weight decay = 0.00001; epoch = 1000.
