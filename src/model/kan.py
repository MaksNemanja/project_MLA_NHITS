import torch
import torch.nn.functional as F
import math
from torch import nn
import numpy as np


def window_functions(x, num_domains, delta=1.9):
    """
    Window functions for finite-basis domain decomposition.
    :param x: input tensor of shape [batch_size, d] where d is 1 or 2.
    :param num_domains: number of domains. If d=2, must be a perfect square.
    :param delta: overlapping ratio. Higher = more overlapping.
    :return: Tensor of shape [batch_size, num_domains] giving the weight of each domain.
    """
    eps = 1e-12

    def w_jl_i(x_i, n_domains, x_min, x_max):
        jvec = torch.arange(n_domains, device=x_i.device) + 1
        muvec = x_min + (jvec - 1) * (x_max - x_min) / (n_domains - 1)
        muvec = muvec.unsqueeze(0).expand(x_i.shape[0], n_domains)
        u = x_i.repeat(1, n_domains)
        sigma = (x_max - x_min) * (delta / 2.0) / (n_domains - 1)

        z = (u - muvec) / (sigma + eps)
        w_jl = ((1 + torch.cos(np.pi * z)) / 2) ** 2
        w_jl = torch.where(torch.abs(z) < 1, w_jl, torch.zeros_like(w_jl))
        return w_jl

    n_dims = x.shape[1]
    if n_dims == 1:
        x_min, x_max = x.min(), x.max()
        w = w_jl_i(x, num_domains, x_min, x_max)
    elif n_dims == 2:
        n_per_dim = int(np.sqrt(num_domains))
        if n_per_dim ** 2 != num_domains:
            raise ValueError("num_domains must be a perfect square for 2D inputs.")
        x_min_x, x_max_x = x[:, 0].min(), x[:, 0].max()
        x_min_y, x_max_y = x[:, 1].min(), x[:, 1].max()
        w1 = w_jl_i(x[:, 0:1], n_per_dim, x_min_x, x_max_x)
        w2 = w_jl_i(x[:, 1:2], n_per_dim, x_min_y, x_max_y)
        w = torch.einsum('bi,bj->bij', w1, w2).reshape(x.shape[0], -1)
        if w.shape[1] > num_domains:
            w = w[:, :num_domains]
    else:
        raise ValueError("Only 1D and 2D inputs are currently supported")

    s = torch.sum(w, dim=1, keepdim=True)
    return w / (s + eps)


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.grid_range = grid_range

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        for layer in self.layers:
            layer.update_grid(x, margin=margin)


class FBKAN(nn.Module):
    """
    Finite-Basis KAN (FBKAN): Gère plusieurs domaines (chaque domaine est un KAN),
    applique window_functions pour pondérer les sorties, et permet une mise à jour
    dynamique de la taille de grille.
    """

    def __init__(
        self,
        insize,
        outsize,
        hsizes=[64],
        num_domains=1,
        grid_sizes=[5],
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        grid_updates=None,
        verbose=False,
    ):
        super().__init__()
        self.in_features = insize
        self.out_features = outsize
        self.num_domains = num_domains
        self.spline_order = spline_order

        self.domain_networks = nn.ModuleList()
        # Chaque domaine est un KAN complet
        for _ in range(num_domains):
            layer_sizes = [insize] + hsizes + [outsize]
            self.domain_networks.append(
                KAN(
                    layers_hidden=layer_sizes,
                    grid_size=grid_sizes[0],
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        self.grid_sizes = grid_sizes
        self.grid_updates = grid_updates or []
        self.current_grid_index = 0
        self.verbose = verbose
        self.enable_standalone_scale_spline = enable_standalone_scale_spline

    def forward(self, x, update_grid=False):
        # Appliquer chaque domaine
        domain_outputs = []
        for domain_kan in self.domain_networks:
            if update_grid:
                domain_kan.update_grid(x)
            domain_outputs.append(domain_kan(x))

        # Stacker les sorties : [batch, num_domains, out_features]
        domain_outputs = torch.stack(domain_outputs, dim=1)

        if self.num_domains == 1:
            # Un seul domaine, pas besoin de fenêtrage
            x_final = domain_outputs.squeeze(1)
        else:
            # Calcul des poids de fenêtrage
            w = window_functions(x, self.num_domains)  # [batch, num_domains]
            w = w.unsqueeze(-1)  # [batch, num_domains, 1]
            # Combiner les sorties
            x_final = torch.sum(w * domain_outputs, dim=1)  # [batch, out_features]

        return x_final

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            domain_kan.regularization_loss(regularize_activation, regularize_entropy)
            for domain_kan in self.domain_networks
        )

    @torch.no_grad()
    def update_grid(self, x, margin=0.01):
        # Mettre à jour les grilles de tous les domaines
        for domain_kan in self.domain_networks:
            domain_kan.update_grid(x, margin=margin)

    @torch.no_grad()
    def update_epoch(self, epoch, x):
        # Si on a défini des moments où l'on change de grid_size, on le fait ici
        if self.current_grid_index < len(self.grid_updates) and epoch >= self.grid_updates[self.current_grid_index]:
            new_grid_size = self.grid_sizes[self.current_grid_index]
            if self.verbose:
                print(f"Updating grid size to {new_grid_size} at epoch {epoch}")

            for domain_kan in self.domain_networks:
                for layer in domain_kan.layers:
                    layer.grid_size = new_grid_size
                    layer.reset_parameters()
                    layer.update_grid(x)
            self.current_grid_index += 1
