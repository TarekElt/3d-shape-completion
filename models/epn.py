

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
	

	def __init__(self, in_ch, out_ch, norm=True):
		super().__init__()
		self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
		self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.norm = norm
		if norm:
			self.bn1 = nn.BatchNorm3d(out_ch)
			self.bn2 = nn.BatchNorm3d(out_ch)
		self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

	def forward(self, x):
		out = self.conv1(x)
		if self.norm:
			out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		if self.norm:
			out = self.bn2(out)
		if self.proj is not None:
			x = self.proj(x)
		return self.relu(out + x)


class UpBlock(nn.Module):
	

	def __init__(self, in_ch, out_ch, res_blocks=1, norm=True):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
		self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
		self.relu = nn.ReLU(inplace=True)
		self.resblocks = nn.Sequential(*[ResBlock3D(out_ch, out_ch, norm=norm) for _ in range(res_blocks)])

	def forward(self, x):
		x = self.up(x)
		x = self.conv(x)
		x = self.relu(x)
		x = self.resblocks(x)
		return x


class EPN_UNet(nn.Module):
	

	def __init__(self, G=32, in_channels=1, base_ch=32, bottleneck_dim=2048,
				 res_blocks=1, use_dropout=False, use_vae=False,
				 conditioning_camera=True, camera_embed_dim=128):
		super().__init__()
		self.G = G
		self.in_channels = in_channels
		self.base_ch = base_ch
		self.bottleneck_dim = bottleneck_dim
		self.res_blocks = res_blocks
		self.use_dropout = use_dropout
		self.use_vae = use_vae
		self.conditioning_camera = conditioning_camera

		# Compute number of downsample levels based on G
		# At least 3 levels (G / 8), and increase for larger G
		self.levels = max(3, int(math.floor(math.log2(max(32, G))) - 2))

		# Build encoder: at each level we double channels and downsample
		self.enc_convs = nn.ModuleList()
		ch = base_ch
		in_ch = in_channels
		for lvl in range(self.levels):
			out_ch = base_ch * (2 ** lvl)
			seq = nn.Sequential(
				nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
				*[ResBlock3D(out_ch, out_ch) for _ in range(self.res_blocks)],
				nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=2, padding=1),
				nn.ReLU(inplace=True),
			)
			self.enc_convs.append(seq)
			in_ch = out_ch

		# Determine spatial size at bottleneck
		down_factor = 2 ** self.levels
		self.bottleneck_s = max(1, int(round(G / down_factor)))
		self.bottleneck_ch = in_ch

		# Bottleneck refinement
		self.bottleneck = nn.Sequential(
			ResBlock3D(self.bottleneck_ch, self.bottleneck_ch),
			ResBlock3D(self.bottleneck_ch, self.bottleneck_ch),
		)

		# Dense projection to latent and back
		feat_flat = self.bottleneck_ch * (self.bottleneck_s ** 3)
		self.fc_mu = nn.Linear(feat_flat, bottleneck_dim)
		if self.use_vae:
			self.fc_logvar = nn.Linear(feat_flat, bottleneck_dim)
		self.fc_decode = nn.Linear(bottleneck_dim, feat_flat)

		# optional dropout
		if use_dropout:
			self.dropout = nn.Dropout(p=0.2)
		else:
			self.dropout = None

		# camera MLP: maps (B,3) -> (B, camera_embed_dim) -> projected to channels
		if conditioning_camera:
			self.camera_mlp = nn.Sequential(
				nn.Linear(3, camera_embed_dim),
				nn.ReLU(inplace=True),
				nn.Linear(camera_embed_dim, camera_embed_dim),
				nn.ReLU(inplace=True),
			)
			self.camera_to_ch = nn.Linear(camera_embed_dim, self.bottleneck_ch)
		else:
			self.camera_mlp = None

		# Decoder: reverse of encoder. Use UpBlock then concat skip then reduce.
		self.up_blocks = nn.ModuleList()
		for lvl in reversed(range(self.levels)):
			out_ch = base_ch * (2 ** lvl)
			self.up_blocks.append(UpBlock(self.bottleneck_ch, out_ch, res_blocks=self.res_blocks))
			self.bottleneck_ch = out_ch

		# Final conv -> 1 channel TSDF output in [-1,1]
		self.final_conv = nn.Sequential(
			nn.Conv3d(base_ch * 2, base_ch, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv3d(base_ch, 1, kernel_size=3, padding=1),
			nn.Tanh(),
		)

	def reparameterize(self, mu, logvar):
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		return mu + eps * std

	def forward(self, x, visibility=None, occupancy=None, camera_dir=None):
		
		# Build input tensor if visibility/occupancy provided
		if x.dim() == 5 and x.size(1) == 1:
			parts = [x]
			if visibility is not None:
				parts.append(visibility)
			if occupancy is not None:
				parts.append(occupancy)
			inp = torch.cat(parts, dim=1) if len(parts) > 1 else x
		else:
			# assume x already contains channels
			inp = x

		# Encoder forward, collect skip connections (pre-downsample)
		skips = []
		h = inp
		for enc in self.enc_convs:
			# run modules in this sequential block and capture skip before downsample
			for module in enc:
				if isinstance(module, nn.Conv3d) and getattr(module, 'stride', (1, 1, 1)) == (2, 2, 2):
					# store skip feature BEFORE downsample
					skips.append(h)
				h = module(h)

		# bottleneck
		b = self.bottleneck(h)
		B = b.shape[0]
		flat = b.view(B, -1)
		mu = self.fc_mu(flat)
		if self.use_vae:
			logvar = self.fc_logvar(flat)
			z = self.reparameterize(mu, logvar)
		else:
			z = mu
		if self.dropout is not None:
			z = self.dropout(z)
		dec_flat = self.fc_decode(z)
		dec = dec_flat.view_as(b)

		# camera conditioning: embed and add as channel bias to dec
		if (camera_dir is not None) and (self.camera_mlp is not None):
			cam = self.camera_mlp(camera_dir)
			cam_ch = self.camera_to_ch(cam).view(B, -1, 1, 1, 1)
			dec = dec + cam_ch

		# Decoder: upsample and merge skips
		current = dec
		for i, up in enumerate(self.up_blocks):
			current = up(current)
			# get corresponding skip (reverse order)
			skip = skips[-(i + 1)]
			# center-crop skip to current size if needed
			if skip.shape[2:] != current.shape[2:]:
				sd, sh, sw = skip.shape[2:]
				cd, ch, cw = current.shape[2:]
				sd0 = max(0, (sd - cd) // 2)
				sh0 = max(0, (sh - ch) // 2)
				sw0 = max(0, (sw - cw) // 2)
				skip = skip[:, :, sd0:sd0 + cd, sh0:sh0 + ch, sw0:sw0 + cw]
			current = torch.cat([current, skip], dim=1)
			# reduce concatenated channels with a 1x1 conv to match up.conv.out_channels
			reduce_conv = nn.Conv3d(current.shape[1], up.conv.out_channels, kernel_size=1).to(current.device)
			current = reduce_conv(current)

		# Ensure expected channels for final conv
		if current.shape[1] != self.final_conv[0].in_channels:
			match_conv = nn.Conv3d(current.shape[1], self.final_conv[0].in_channels, kernel_size=1).to(current.device)
			current = match_conv(current)

		out = self.final_conv(current)
		return out


def get_model(G=32, mode='unet', conditioning=True, size='medium'):
	
	size_map = {'small': 512, 'medium': 2048, 'large': 8192}
	if size not in size_map:
		raise ValueError('size must be small/medium/large')
	bottleneck = size_map[size]
	base_ch = 16 if size == 'small' else 32 if size == 'medium' else 48
	if mode == 'unet':
		return EPN_UNet(G=G, in_channels=1, base_ch=base_ch, bottleneck_dim=bottleneck,
						 res_blocks=1, use_dropout=(size == 'small'), use_vae=False,
						 conditioning_camera=conditioning)
	else:
		raise ValueError('Unknown mode: ' + str(mode))

