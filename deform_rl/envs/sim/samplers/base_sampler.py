class BaseSampler:
    def __init__(self):
        pass

    def sample(self,*args):
        raise NotImplementedError

    def analytics(self):
        return None

