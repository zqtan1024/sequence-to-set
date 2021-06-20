import pdb
import torch
from ssn import util


def create_train_sample(doc):
    
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    pos_encoding = [t.pos_id for t in doc.tokens]
    wordvec_encoding = [t.wordinx for t in doc.tokens]
    
    # all tokens
    token_masks = []
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))

    gt_entities_spans_token = []
    gt_entity_types = []
    gt_entity_masks = []
    for e in doc.entities:
        gt_entities_spans_token.append(e.span_token)
        gt_entity_types.append(e.entity_type.index)
        gt_entity_masks.append(1)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    if len(gt_entity_types) > 0:
        gold_entity_types = torch.tensor(gt_entity_types, dtype=torch.long)
        gold_entity_spans_token = torch.tensor(gt_entities_spans_token, dtype=torch.long)
        gold_entity_spans_token[:, 1] = gold_entity_spans_token[:, 1] - 1
        gold_entity_masks = torch.tensor(gt_entity_masks, dtype=torch.bool)
    else:
        gold_entity_types = torch.zeros([1], dtype=torch.long)
        gold_entity_spans_token = torch.zeros([1, 2], dtype=torch.long)
        gold_entity_masks = torch.zeros([1], dtype=torch.bool)
    
    
    return dict(encodings=encodings, context_masks=context_masks, token_masks_bool=token_masks_bool, token_masks=token_masks, 
                gold_entity_types=gold_entity_types, gold_entity_spans_token=gold_entity_spans_token, gold_entity_masks=gold_entity_masks,
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, 
                char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count)


def create_eval_sample(doc):

    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    char_encodings = doc.char_encoding
    char_encoding = []
    char_count = []
    for char_encoding_token in char_encodings:
        char_count.append(len(char_encoding_token))
        char_encoding.append(torch.tensor(char_encoding_token,dtype=torch.long))
    char_encoding = util.padded_stack(char_encoding)
    token_masks_char = (char_encoding!=0).long()
    char_count = torch.tensor(char_count, dtype = torch.long)

    pos_encoding = [t.pos_id for t in doc.tokens]
    wordvec_encoding = [t.wordinx for t in doc.tokens]

    # all tokens
    token_masks = []
    for t in doc.tokens:
        token_masks.append(create_entity_mask(*t.span, context_size))

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)
    pos_encoding = torch.tensor(pos_encoding, dtype=torch.long)
    wordvec_encoding = torch.tensor(wordvec_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)
    token_masks_bool = torch.ones(token_count, dtype=torch.bool)
    token_masks = torch.stack(token_masks)

    return dict(encodings=encodings, context_masks=context_masks, token_masks_bool=token_masks_bool,token_masks=token_masks, 
                pos_encoding = pos_encoding, wordvec_encoding = wordvec_encoding, 
                char_encoding = char_encoding, token_masks_char = token_masks_char, char_count = char_count)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
