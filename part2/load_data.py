import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
START_IDX = 32099

SQL_QUOTED = re.compile(r"""'(?:''|[^'])*'|"(?:""|[^"])*"|`[^`]*`|\[[^\]]*\]""")
SQL_FROM = re.compile(
    r"\bFROM\s+([^();]+?)(?=\s+(?:WHERE|GROUP\s+BY|ORDER\s+BY|HAVING|LIMIT|"
    r"UNION|INTERSECT|EXCEPT)\b|[);]|$)",
    re.IGNORECASE,
)
TABLE_SOURCE = re.compile(r"\s*([a-z_]\w*)(\s+(?:AS\s+)?([a-z_]\w*))?\s*", re.IGNORECASE)
QUALIFIER = re.compile(r"\b([a-z_]\w*)(?=\s*\.)", re.IGNORECASE)


def remove_redundant_aliases(sql):
    # Parse the comma-separated FROM lists used by the assignment. Mask strings
    # without changing character offsets, so edits never touch quoted values.
    masked = SQL_QUOTED.sub(lambda match: " " * len(match.group()), sql)
    from_lists = list(SQL_FROM.finditer(masked))
    if len(from_lists) != len(re.findall(r"\bFROM\b", masked, re.IGNORECASE)):
        return sql

    tables = []
    for from_list in from_lists:
        offset = from_list.start(1)
        for source in from_list.group(1).split(","):
            match = TABLE_SOURCE.fullmatch(source)
            if match is None:
                return sql
            tables.append((match, offset))
            offset += len(source) + 1

    table_counts = Counter(match[1].lower() for match, _ in tables)
    names = Counter((match[3] or match[1]).lower() for match, _ in tables)
    replacements = {}
    edits = []
    for match, offset in tables:
        table, alias = match[1], match[3]
        if alias is None or table_counts[table.lower()] != 1 or names[alias.lower()] != 1:
            continue
        if alias.lower() != table.lower() and names[table.lower()]:
            continue
        replacements[alias.lower()] = table
        edits.append((offset + match.start(2), offset + match.end(2), ""))

    for qualifier in QUALIFIER.finditer(masked):
        table = replacements.get(qualifier[1].lower())
        if table is not None:
            edits.append((*qualifier.span(1), table))

    for start, end, replacement in sorted(edits, reverse=True):
        sql = sql[:start] + replacement + sql[end:]
    return sql


class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.data = self.process_data(data_folder, split, self.tokenizer)
        

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        nl = load_lines(data_folder+'/'+split+'.nl')

        tokenized_nl = tokenizer(nl, padding = False)
        encoded_nl = tokenized_nl['input_ids']
                
        if split != "test":
            sql = load_lines(data_folder+'/'+split+'.sql')
            sql = [remove_redundant_aliases(query) for query in sql]
            tokenized_sql = tokenizer(sql, padding = False)
            encoded_sql = tokenized_sql['input_ids']
            
            if len(encoded_nl) == len(encoded_sql):
                return list(zip(encoded_nl, encoded_sql))
            else:
                raise ValueError('Your nl and sql files are of different length.')

        return encoded_nl

    def __len__(self):
        # TODO
        return len(self.data)
    
    def __getitem__(self, idx):
        # TODO
        return self.data[idx] 

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO

    encoder_ids = [torch.tensor(nl, dtype = torch.long) for nl, sql in batch]
    padded_encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value = PAD_IDX)
    encoder_mask = (padded_encoder_ids != PAD_IDX).long()


    decoder_inputs = [torch.tensor([START_IDX] + sql[:-1], dtype = torch.long) for nl, sql in batch]
    padded_decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value= PAD_IDX)
    decoder_targets = [torch.tensor(sql, dtype = torch.long) for nl, sql in batch]
    padded_decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value= PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), START_IDX, dtype = torch.long)

    return padded_encoder_ids, encoder_mask, padded_decoder_inputs, padded_decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids = [torch.tensor(nl, dtype=torch.long) for nl in batch]
    padded_encoder_ids = pad_sequence(encoder_ids, batch_first = True, padding_value = PAD_IDX)
    encoder_mask = (padded_encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), START_IDX, dtype = torch.long)

    return padded_encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x