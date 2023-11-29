import torch
A = torch.randn(16, 512)
B = torch.randn(16, 1024)
C = torch.randn(16, 32)

n = A.shape[0]
# 用 1 扩充维度
A = torch.cat([A, torch.ones(n, 1)], dim=1)
B = torch.cat([B, torch.ones(n, 1)], dim=1)
C = torch.cat([C, torch.ones(n, 1)], dim=1)

A = A.unsqueeze(2)  # [n, A, 1]
B = B.unsqueeze(1)  # [n, 1, B]
fusion_AB = torch.einsum('nxt, nty->nxy', A, B)  # [n, A, B]
fusion_AB = fusion_AB.flatten(start_dim=1).unsqueeze(1) # [n, AxB, 1]
C = C.unsqueeze(1) # [n, 1, C]
fusion_ABC = torch.einsum('ntx, nty->nxy', fusion_AB, C) # [n, AxB, C]
fusion_ABC = fusion_ABC.flatten(start_dim=1)  # [n, AxBxC]

