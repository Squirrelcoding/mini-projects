import numpy as np
from scipy.stats import t

np.random.seed(42)

ALPHA = 0.05

class SimpleLinear():
    def __init__(self, X: np.array, y: np.array):
        assert X.shape == y.shape
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.params = {}
    
    def fit(self):

        # Estimate parameters
        b1_hat = (self.X - self.X.mean()).dot(self.y - self.y.mean()) / ((self.X - self.X.mean()).dot(self.X - self.X.mean()))
        b0_hat = self.y.mean() - b1_hat * self.X.mean()

        # Test values
        y_hat = b0_hat + b1_hat * self.X

        # Variance estimate
        var_hat = ((self.y - y_hat).dot(self.y - y_hat)) / (self.X.shape[0] - 2)

        # Other statistics
        sse = (self.y - y_hat).dot(self.y - y_hat)
        sst = (self.y - self.y.mean()).dot(self.y - self.y.mean())
        ssr = (y_hat - self.y.mean()).dot(y_hat - self.y.mean())

        # R^2 value
        r_squared = 1 - sse / sst

        # Estimate variance of b1
        b1_var = var_hat / ((self.X - self.X.mean()).dot(self.X - self.X.mean()))

        print(t.ppf(q=1 - ALPHA / 2, df=self.N - 2))

        zero_t_b_1stat = b1_hat / b1_var 

        self.params = {
            "b0_hat": b0_hat,
            "b1_hat": {
                "val": b1_hat,
                "variance": b1_var,
                "ci": (
                    b1_hat - t.ppf(q=1 - ALPHA / 2, df=self.N - 2) * np.sqrt(b1_var), 
                    b1_hat + t.ppf(q=1 - ALPHA / 2, df=self.N - 2) * np.sqrt(b1_var)
                ),
                "t_stat": t.ppf(q=1- ALPHA / 2, df=self.N - 2),
                "p": 2 * (1 - t.cdf(abs(zero_t_b_1stat), self.N - 2))
            },
            "var_hat": var_hat,
            "r_squared": r_squared
        }
    
    def predict(self, x):
        y_hat = self.params.b0_hat * x + self.params.b1_hat

        se = self.params.var_hat * np.sqrt(1.0 + 1.0 / self.N + ((x - self.X.mean()) * (x - self.X.mean())) / (self.X - self.X.mean()) * (self.X - self.X.mean()))

        t_stat = t.ppf(q=1 - ALPHA/2, df=self.N - 2)

        ci = (y_hat - se * t_stat, y_hat + se * t_stat)

        return (y_hat, ci)

    def corr(self):
        """
        Estimate of correlation coefficient
        """
        s_xy = (self.X - self.X.mean()).dot(self.y - self.y.mean())

        p_hat = s_xy / (np.sqrt( (self.X - self.X.mean()).dot(self.X - self.X.mean()) ) * np.sqrt( (self.y - self.y.mean()).dot(self.y - self.y.mean()) ))

        t_stat = p_hat * np.sqrt(self.N - 2) / np.sqrt(1 - p_hat * p_hat)

        return p_hat, t_stat

# Exact copies. Very low variance.

X = np.linspace(0, 1, 50)
y = 3.0 * X + 2.0

model = SimpleLinear(X, y)
model.fit()

# [np.float64(2.0), np.float64(2.9999999999999996), np.float64(1.355854680848614e-31)]
print(model.params)

# There's a divide by zero since there is super low variance and very strong correlation coefficient
print(model.corr())

y = np.random.normal(3.0 * X + 2.0, 0.3)

model = SimpleLinear(X, y)
model.fit()

# [np.float64(2.0), np.float64(2.9999999999999996), np.float64(1.355854680848614e-31)]
print(model.params)
print(model.corr())
