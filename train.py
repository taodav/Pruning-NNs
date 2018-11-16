class SparseTrainer:
    def __init__(self, model, train_dataset, valid_dataset, args):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.args = args