import torch.legacy.nn as nn

# Modelo da rede neural
def FCNN():
	num_classes = 2
	n_layers_enc = 32
	n_layers_ctx = 128
	n_input = 5
	prob_drop = 0.25
	layers = []
	# Encoder
	model = nn.Sequential()
	pool = nn.SpatialMaxPooling(2,2,2,2)
	model.add(nn.SpatialConvolution(n_input, n_layers_enc, 3, 3, 1, 1, 1, 1))
	model.add(nn.ELU())
	model.add(nn.SpatialConvolution(n_layers_enc, n_layers_enc, 3, 3, 1, 1, 1, 1))
	model.add(nn.ELU())
	model.add(pool)
	# Context Module
	model.add(nn.SpatialDilatedConvolution(n_layers_enc, n_layers_ctx, 3, 3, 1, 1, 1, 1, 1, 1))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 2, 2, 2, 2))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 4, 4, 4, 4))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 8, 8, 8, 8))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 16, 16, 16, 16))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 32, 32, 32, 32))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_ctx, 3, 3, 1, 1, 64, 64, 64, 64))
	model.add(nn.ELU())
	model.add(nn.SpatialDropout(prob_drop))
	model.add(nn.SpatialDilatedConvolution(n_layers_ctx, n_layers_enc, 1, 1))
	model.add(nn.ELU())	# Nao havia no paper
	# Decoder
	model.add(nn.SpatialMaxUnpooling(pool))
	model.add(nn.SpatialConvolution(n_layers_enc, n_layers_enc, 3, 3, 1, 1, 1, 1))
	model.add(nn.ELU())
	model.add(nn.SpatialConvolution(n_layers_enc, num_classes, 3, 3, 1, 1, 1, 1))
	model.add(nn.ELU())
	model.add(nn.SoftMax()) # Nao havia no paper
	return model
