from collections import OrderedDict
from typing import List
from torch.utils.data import Dataset as TorchDataset

from ssn import sampling


class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)

    def __str__(self) -> str:
        return self._identifier + "=" + self._verbose_name


class Token:
    def __init__(self, tid: int, index: int, span_start: int, span_end: int, phrase: str, pos: str, vocab_id: int, char_start: int, char_end: int):
        self._tid = tid  # ID within the corresponding dataset
        self._index = index  # original token index in document

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase
        self._pos = pos
        self._char_start = char_start
        self._char_end = char_end
        self._vocab_id = vocab_id

    @property
    def index(self):
        return self._index
    
    @property
    def wordinx(self):
        return self._vocab_id

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def char_start(self):
        return self._char_start

    @property
    def char_end(self):
        return self._char_end

    @property
    def char_span(self):
        return self._char_start, self._char_end

    @property
    def phrase(self):
        return self._phrase

    @property
    def pos(self):
        return self._pos
    
    @property
    def pos_id(self):
        return self._pos

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self._tokens[0].index,self._tokens[-1].index + 1

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __str__(self) -> str:
        return " ".join([str(t) for t in self._tokens])


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, tokens: List[Token], phrase: str):
        self._eid = eid  # ID within the corresponding dataset

        self._entity_type = entity_type

        self._tokens = tokens
        self._phrase = phrase

    def as_tuple(self):
        return self.span_start, self.span_end, self._entity_type

    def as_tuple_token(self):
        return self._tokens[0].index,self._tokens[-1].index+1, self._entity_type

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def span_token(self):
        return self._tokens[0].index,self._tokens[-1].index+1

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_entity: Entity,
                 tail_entity: Entity, reverse: bool = False):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._reverse = reverse

        self._first_entity = head_entity if not reverse else tail_entity
        self._second_entity = tail_entity if not reverse else head_entity

    def as_tuple(self):
        head = self._head_entity
        tail = self._tail_entity
        head_start, head_end = (head.span_start, head.span_end)
        tail_start, tail_end = (tail.span_start, tail.span_end)

        t = ((head_start, head_end, head.entity_type),
             (tail_start, tail_end, tail.entity_type), self._relation_type)
        return t

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    @property
    def first_entity(self):
        return self._first_entity

    @property
    def second_entity(self):
        return self._second_entity

    @property
    def reverse(self):
        return self._reverse

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(self, doc_id: int, tokens: List[Token], entities: List[Entity],
                 encoding: List[int], char_encoding: List[List[int]]):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._tokens = tokens
        self._entities = entities

        # byte-pair document encoding including special tokens ([CLS] and [SEP])
        self._encoding = encoding
        self._char_encoding = char_encoding

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def entities(self):
        return self._entities

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encoding(self):
        return self._encoding

    @property
    def char_encoding(self):
        return self._char_encoding


    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @char_encoding.setter
    def char_encoding(self, value):
        self._char_encoding = value

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)


class BatchIterator:
    def __init__(self, entities, batch_size, order=None, truncate=False):
        self._entities = entities
        self._batch_size = batch_size
        self._truncate = truncate
        self._length = len(self._entities)
        self._order = order

        if order is None:
            self._order = list(range(len(self._entities)))

        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._truncate and self._i + self._batch_size > self._length:
            raise StopIteration
        elif not self._truncate and self._i >= self._length:
            raise StopIteration
        else:
            entities = [self._entities[n] for n in self._order[self._i:self._i + self._batch_size]]
            self._i += self._batch_size
            return entities


class Dataset(TorchDataset):
    TRAIN_MODE = 'train'
    EVAL_MODE = 'eval'

    def __init__(self, label, rel_types, entity_types):
        self._label = label
        self._rel_types = rel_types
        self._entity_types = entity_types
        self._mode = Dataset.TRAIN_MODE

        self._documents = OrderedDict()
        self._entities = OrderedDict()

        # current ids
        self._doc_id = 0
        self._rid = 0
        self._eid = 0
        self._tid = 0

    def iterate_documents(self, batch_size, order=None, truncate=False):
        return BatchIterator(self.documents, batch_size, order=order, truncate=truncate)

    def create_token(self, idx, span_start, span_end, phrase, pos, inx, char_start, char_end) -> Token:
        token = Token(self._tid, idx, span_start, span_end, phrase, pos, inx, char_start, char_end)

        self._tid += 1
        return token

    def create_document(self, tokens, entity_mentions, doc_encoding, char_encoding) -> Document:
        document = Document(self._doc_id, tokens, entity_mentions, doc_encoding, char_encoding)

        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def create_entity(self, entity_type, tokens, phrase) -> Entity:
        mention = Entity(self._eid, entity_type, tokens, phrase)
        self._entities[self._eid] = mention
        self._eid += 1
        return mention

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == Dataset.TRAIN_MODE:
            return sampling.create_train_sample(doc)
        else:
            return sampling.create_eval_sample(doc)

    def switch_mode(self, mode):
        self._mode = mode

    @property
    def label(self):
        return self._label

    @property
    def input_reader(self):
        return self._input_reader

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entities)

