


class ContinuationSchedule:
    """Gradually increase alpha_max and beta over optimization iterations."""
    
    def __init__(self, alpha_start=5.0, alpha_end=100.0, alpha_growth_rate=1.005,
                 beta_start=1.0, beta_end=32.0, beta_growth_rate=1.005,
                 beta_delay=400):
        """
        alpha_growth_rate/beta_growth_rate: multiplicative each iteration
        beta_delay: hold beta at beta_start for this many iterations
        """
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_growth = alpha_growth_rate
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_growth = beta_growth_rate
        self.beta_delay = beta_delay
    
    def alpha(self, iteration):
        """Alpha_max at given iteration."""
        raw = self.alpha_start * self.alpha_growth ** iteration
        return min(raw, self.alpha_end)
    
    def beta(self, iteration):
        """Beta at given iteration; held at beta_start until delay."""
        if iteration < self.beta_delay:
            return self.beta_start
        raw = self.beta_start * self.beta_growth ** (iteration - self.beta_delay)
        return min(raw, self.beta_end)