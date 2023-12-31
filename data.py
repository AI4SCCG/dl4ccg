from torch.utils.data import Dataset

import utils


class CodePtrDataset(Dataset):

    def __init__(self, source_path, code_path, ast_path, nl_path):
        # get lines
        sources = utils.load_dataset(source_path)
        codes = utils.load_dataset(code_path)
        asts = utils.load_dataset(ast_path)
        nls = utils.load_dataset(nl_path)

        assert len(sources) == len(codes) == len(asts) == len(nls)


        self.sources, self.codes, self.asts, self.nls = sources, codes, asts, nls

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, index):
        return self.sources[index], self.codes[index], self.asts[index], self.nls[index]

    def get_dataset(self):
        return self.sources, self.codes, self.asts, self.nls
