class biasRV():
    def __init__(self, predict, X, Y, alpha):
        self.predict = predict
        self.X = X
        self.Y = Y
        self.alpha = alpha

    def set_predictor(self, predict):
        raise NotImplementedError
    
    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def set_alpha(self, alpha):
        self.alpha = alpha
        
        