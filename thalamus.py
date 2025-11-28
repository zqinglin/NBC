import torch
import math


class KAB_Thalamus:
    def __init__(self, slots, embedding_dim, anchors=None):
        self.slots = list(slots)
        self.embedding_dim = int(embedding_dim)
        self.alpha = {s: torch.tensor(1.0) for s in self.slots}
        self.beta = {s: torch.tensor(1.0) for s in self.slots}
        if anchors is None:
            anchors = {s: torch.randn(self.embedding_dim) for s in self.slots}
        self.anchors = {s: self._normalize(v.detach().cpu().float()) for s, v in anchors.items()}

    def _normalize(self, v):
        n = v.norm(p=2)
        if n.item() == 0.0:
            return v
        return v / n

    def _cosine_distance(self, x, y):
        x = self._normalize(x)
        y = self._normalize(y)
        return 1.0 - torch.dot(x, y).clamp(-1.0, 1.0)

    def route(self, input_embedding):
        x = input_embedding.detach().cpu().float().view(-1)
        best_slot = None
        best_score = -1e9
        for s in self.slots:
            a = float(self.alpha[s].item())
            b = float(self.beta[s].item())
            dist = float(self._cosine_distance(x, self.anchors[s]).item())
            sample = torch.distributions.Beta(a, b).sample().item()
            score = sample * math.exp(-dist)
            if score > best_score:
                best_score = score
                best_slot = s
        return best_slot

    def update_belief(self, slot, reward):
        r = float(max(0.0, min(1.0, reward)))
        self.alpha[slot] = self.alpha[slot] + torch.tensor(r)
        self.beta[slot] = self.beta[slot] + torch.tensor(1.0 - r)

    def set_anchor(self, slot, vector):
        self.anchors[slot] = self._normalize(vector.detach().cpu().float().view(-1))

