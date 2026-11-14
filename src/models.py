import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TreeGram(nn.Module):
    dynamic = False

    def __init__(self, vocab_size, n, d, dff=4.0, mu=1.0, min_dim=16):
        super().__init__()
        assert n >= 3, "the context length n must be at least 3"
        self.n = n
        self.d = d
        self.vocab_size = vocab_size
        self.dff = dff
        self.mu = mu
        self.depth = self.find_depth(n)

        self.emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(n - 1, d)

        if self.dynamic:
            self.routers = nn.ModuleList()
        self.W_up = nn.ParameterList()
        self.W_down = nn.ParameterList()
        if dff > 0:
            self.mlp_up = nn.ModuleList()

        nl = n
        for i in range(self.depth - 1):
            g = self.num_merges(nl)
            if self.dynamic:
                self.routers.append(nn.Linear(d, g, bias=False))
            self.register_buffer(f"topo_map_{i}", torch.tensor([self.return_idx(k, nl) for k in range(g)], dtype=torch.long))
            if self.dynamic:
                self.register_buffer(f"router_bias_{i}", torch.zeros(g))
                target = torch.zeros(g)
                if self.noisy_tgt > 0.0:
                    target = target + self.noisy_tgt * torch.randn(g)
                self.register_buffer(f"router_target_{i}", torch.softmax(target, dim=0))

            nl_next = nl // 2 + 1
            W = torch.empty(nl_next - 1, d, d)
            nn.init.xavier_normal_(W)
            self.W_up.append(nn.Parameter(W))
            W = torch.empty(d, d)
            nn.init.xavier_normal_(W)
            self.W_down.append(nn.Parameter(W))
            if dff > 0:
                self.mlp_up.append(MLP(d, max(min_dim, int(dff * d)), d))
            nl = nl_next

        self.W_top = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_normal_(self.W_top)
        if dff > 0:
            self.mlp_top = MLP(d, max(min_dim, int(dff * d)), d)
            self.mlp_down = nn.ModuleList([MLP(d, max(min_dim, int(dff * d)), d) for _ in range(self.depth - 1)])

        if self.dynamic:
            self.norm_router = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(self.depth - 1)])
        self.norm_up_proj = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(self.depth - 1)])
        self.norm_top_proj = nn.LayerNorm(d, bias=False)
        self.norm_down_proj = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(self.depth - 1)])
        self.norm_final = nn.LayerNorm(d, bias=False)
        if dff > 0:
            self.norm_up_mlp = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(self.depth - 1)])
            self.norm_top_mlp = nn.LayerNorm(d, bias=False)
            self.norm_down_mlp = nn.ModuleList([nn.LayerNorm(d, bias=False) for _ in range(self.depth - 1)])

        T = n - 1
        src = torch.arange(T).view(1, T) - (T - 1 - torch.arange(T).view(T, 1))
        self.register_buffer("ar_prefix_mask", src >= 0)
        self.register_buffer("ar_prefix_src_idx", src.clamp(min=0))

        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
        if self.dynamic:
            for router in self.routers:
                nn.init.normal_(router.weight, mean=0.0, std=0.02)

    @staticmethod
    def find_depth(n):
        depth = 1
        while n > 2:
            n = n // 2 + 1
            depth += 1
        return depth

    @staticmethod
    def num_merges(nl):
        return nl // 2 + 1 if nl % 2 == 1 else nl * (nl // 2 + 1) // 4

    @staticmethod
    def return_idx(idx, nl):
        if nl % 2 == 1:
            return [2 * idx]
        k = nl // 2
        rem = idx
        for ai in range(k):
            if rem < k - ai:
                return [2 * ai, 2 * ai + 1 + 2 * rem]
            rem -= k - ai

    def embed(self, X):
        B, T = X.shape
        assert T == self.n - 1, "expected inputs of shape (B, n-1)"
        X_emb = self.emb(X) + self.pos_emb(torch.arange(T, device=X.device))
        d = X_emb.shape[-1]
        gather_idx = self.ar_prefix_src_idx.view(1, T, T, 1).expand(B, T, T, d)
        Z = X_emb.unsqueeze(1).expand(B, T, T, d).gather(dim=2, index=gather_idx)
        Z = Z * self.ar_prefix_mask.view(1, T, T, 1).to(dtype=Z.dtype)
        active = self.ar_prefix_mask.view(1, T, T).expand(B, T, T).reshape(B * T, T)
        pad = (~self.ar_prefix_mask).view(1, T, T).expand(B, T, T).reshape(B * T, T)
        return Z.reshape(B * T, T, d), active, pad, active

    @staticmethod
    def insert(X, pos, fill_val):
        B, n = X.shape[:2]
        _, m, k = pos.shape
        out = torch.full((B, m, n + k) + tuple(X.shape[2:]), fill_val, device=X.device, dtype=X.dtype)
        idx = torch.arange(n, device=X.device).view(1, 1, n, 1)
        new_idx = idx.squeeze(-1) + (idx >= pos.unsqueeze(2)).sum(dim=-1)
        b = torch.arange(B, device=X.device).view(B, 1, 1).expand(B, m, n)
        m_idx = torch.arange(m, device=X.device).view(1, m, 1).expand(B, m, n)
        out[b, m_idx, new_idx] = X.unsqueeze(1).expand((B, m) + tuple(X.shape[1:]))
        return out

    def merge_level(self, i, Z, active, pad, pos):
        Z = self.norm_up_proj[i](Z)
        z_mod = self.insert(Z, pos, 1.0)
        active_mod = self.insert(active, pos, False)
        pad_mod = self.insert(pad, pos, False)
        on_mod = active_mod | (~pad_mod)

        last = z_mod[:, :, -1, :]
        last_active = active_mod[:, :, -1]
        last_pad = pad_mod[:, :, -1]
        last_on = last_active | (~last_pad)
        last = torch.where(last_on.unsqueeze(-1), last, torch.ones_like(last))

        pair_active = active_mod[:, :, 0:-2:2] | active_mod[:, :, 1:-1:2]
        pair_pad = pad_mod[:, :, 0:-2:2] | pad_mod[:, :, 1:-1:2]
        pair_on = pair_active | (~pair_pad)

        z_res = torch.where(on_mod.unsqueeze(-1), self.insert(Z, pos, 0.0), torch.zeros_like(z_mod))
        res_last = z_res[:, :, -1, :]
        res = 0.5 * (z_res[:, :, 0:-2:2, :] + z_res[:, :, 1:-1:2, :])

        z_mod = torch.where(on_mod.unsqueeze(-1), z_mod, torch.ones_like(z_mod))
        p = torch.einsum('bmnd,ndr->bmnr', z_mod[:, :, 0:-2:2, :] * z_mod[:, :, 1:-1:2, :], self.W_up[i])
        p = p + self.mu * res
        p = p * pair_on.unsqueeze(-1).to(dtype=p.dtype)
        return p, pair_active, pair_pad, pair_on, last, last_active, last_pad, last_on, res_last

    def mlp(self, x, norm, mlp):
        if self.dff > 0:
            return mlp(norm(x)) + self.mu * x
        return x

    def top_and_down(self, Z, active, pad, on, lasts, last_active, last_pad, last_on, res_lasts):
        Z_pre = self.norm_top_proj(Z) @ self.W_top + self.mu * Z
        Z_pre = Z_pre * on.unsqueeze(-1).to(dtype=Z_pre.dtype)
        Z = self.mlp(Z_pre, getattr(self, "norm_top_mlp", None), getattr(self, "mlp_top", None))
        Z = Z * on.unsqueeze(-1).to(dtype=Z.dtype)

        for i in range(self.depth - 2, -1, -1):
            Z = self.norm_down_proj[i](Z)
            Z_in = torch.where(on.unsqueeze(-1), Z, torch.ones_like(Z))
            last = torch.where(last_on[i].unsqueeze(-1), lasts[i], torch.ones_like(lasts[i]))
            Z_pre = last * (Z_in @ self.W_down[i].transpose(0, 1))
            active = active | last_active[i]
            pad = pad | last_pad[i]
            on = active | (~pad)
            Z_pre = Z_pre + self.mu * res_lasts[i] * 0.5
            Z_pre = Z_pre * on.unsqueeze(-1).to(dtype=Z_pre.dtype)
            Z = self.mlp(Z_pre, self.norm_down_mlp[i] if self.dff > 0 else None,
                         self.mlp_down[i] if self.dff > 0 else None)
            Z = Z * on.unsqueeze(-1).to(dtype=Z.dtype)

        return self.norm_final(Z) @ self.emb.weight.T


class FixedTree(TreeGram):
    def __init__(self, vocab_size, n, d, topo=0, dff=4.0, mu=1.0):
        super().__init__(vocab_size, n, d, dff=dff, mu=mu)
        sizes = []
        nl = n
        for _ in range(self.depth - 1):
            sizes.append(self.num_merges(nl))
            nl = nl // 2 + 1
        total = 1
        for s in sizes:
            total *= s
        if not -total <= topo < total:
            raise ValueError(f"topo must lie in [-{total}, {total})")
        topo %= total
        idxs = []
        for s in reversed(sizes):
            idxs.append(topo % s)
            topo //= s
        self.register_buffer("fixed_topo_idxs", torch.tensor(idxs[::-1], dtype=torch.long))

    def forward(self, X):
        B0, T = X.shape
        Z, active, pad, on = self.embed(X)
        B = Z.shape[0]
        lasts, last_active, last_pad, last_on, res_lasts = [], [], [], [], []

        for i in range(self.depth - 1):
            topo_map = getattr(self, f"topo_map_{i}")
            pos = topo_map[self.fixed_topo_idxs[i]].view(1, 1, -1).expand(B, 1, -1)
            p, pair_active, pair_pad, pair_on, last, l_act, l_pad, l_on, r_last = self.merge_level(i, Z, active, pad, pos)
            Z = self.mlp(p, self.norm_up_mlp[i] if self.dff > 0 else None, self.mlp_up[i] if self.dff > 0 else None)
            Z = (Z * pair_on.unsqueeze(-1).to(dtype=Z.dtype)).squeeze(1)
            active, pad, on = pair_active.squeeze(1), pair_pad.squeeze(1), pair_on.squeeze(1)
            lasts.append(last.squeeze(1)); last_active.append(l_act.squeeze(1))
            last_pad.append(l_pad.squeeze(1)); last_on.append(l_on.squeeze(1)); res_lasts.append(r_last.squeeze(1))

        logits = self.top_and_down(Z.squeeze(1), active.squeeze(1), pad.squeeze(1), on.squeeze(1),
                                   lasts, last_active, last_pad, last_on, res_lasts)
        return logits.view(B0, T, self.vocab_size)


class DynamicTree(TreeGram):
    dynamic = True

    def __init__(self, vocab_size, n, d, m=1, dff=4.0, mu=1.0, beta=1.0, noisy_tgt=0.1, router_bias_lr=1e-3):
        self.m = m
        self.beta = beta
        self.noisy_tgt = noisy_tgt
        self.router_bias_lr = router_bias_lr
        super().__init__(vocab_size, n, d, dff=dff, mu=mu)

    def route(self, i, Z, on):
        logits = self.routers[i](self.norm_router[i](Z))
        mask = on.unsqueeze(-1).to(dtype=logits.dtype)
        logits *= mask
        raw = logits.sum(dim=-2) / mask.sum(dim=-2).clamp_min(1.0)
        raw_probs = F.softmax(self.beta * raw, dim=-1)
        biased_probs = F.softmax(self.beta * raw + getattr(self, f"router_bias_{i}"), dim=-1)
        g = raw.shape[-1]
        k = min(self.m, g)
        _, top_idx = torch.topk(biased_probs, k=k, dim=-1)
        weights = torch.gather(raw_probs, dim=-1, index=top_idx)
        chosen = F.one_hot(top_idx, num_classes=g).sum(dim=1).to(raw_probs.dtype)
        self._router_frac_selected.append((chosen.mean(dim=0) / k).detach())
        return top_idx, weights

    def forward(self, X):
        B0, T = X.shape
        self._router_frac_selected = []
        Z, active, pad, on = self.embed(X)
        lasts, last_active, last_pad, last_on, res_lasts = [], [], [], [], []

        for i in range(self.depth - 1):
            top_idx, w = self.route(i, Z, on)
            pos = getattr(self, f"topo_map_{i}")[top_idx]
            p, pair_active, pair_pad, pair_on, last, l_act, l_pad, l_on, r_last = self.merge_level(i, Z, active, pad, pos)
            Z_pre = (w.unsqueeze(-1).unsqueeze(-1) * p).sum(dim=1)
            has_on = pair_on.any(dim=1)
            active = (pair_active & pair_on).any(dim=1)
            pad = (pair_pad & pair_on).any(dim=1) | (~has_on)
            on = active | (~pad)
            Z_pre = Z_pre * on.unsqueeze(-1).to(dtype=Z_pre.dtype)
            Z = self.mlp(Z_pre, self.norm_up_mlp[i] if self.dff > 0 else None, self.mlp_up[i] if self.dff > 0 else None)
            Z = Z * on.unsqueeze(-1).to(dtype=Z.dtype)

            lasts.append((w.unsqueeze(-1) * last).sum(dim=1))
            has_on = l_on.any(dim=1)
            l_active = (l_act & l_on).any(dim=1)
            l_padded = (l_pad & l_on).any(dim=1) | (~has_on)
            last_active.append(l_active); last_pad.append(l_padded); last_on.append(l_active | (~l_padded))
            res_lasts.append((w.unsqueeze(-1) * r_last).sum(dim=1))

        logits = self.top_and_down(Z.squeeze(1), active.squeeze(1), pad.squeeze(1), on.squeeze(1),
                                   lasts, last_active, last_pad, last_on, res_lasts)
        return logits.view(B0, T, self.vocab_size)

    @torch.no_grad()
    def update_router_biases(self):
        for i, frac in enumerate(self._router_frac_selected):
            err = frac - getattr(self, f"router_target_{i}")
            getattr(self, f"router_bias_{i}").add_(-self.router_bias_lr * torch.sign(err))
