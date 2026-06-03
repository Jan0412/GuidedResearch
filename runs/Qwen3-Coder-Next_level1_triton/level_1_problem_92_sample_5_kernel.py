exclusive_cumsum = torch.cat((torch.zeros_like(x.select(self.dim, 0).unsqueeze(self.dim)), x), dim=self.dim)[:-1]
return torch.cumsum(exclusive_cumsum, dim=self.dim)