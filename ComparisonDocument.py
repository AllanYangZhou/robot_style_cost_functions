from mongoengine import (
    Document, StringField, IntField,
    BinaryField
)
from bson.binary import Binary
import pickle


def array_to_binary(x):
    '''Numpy array to bson binary'''
    return Binary(pickle.dumps(x))


def binary_to_array(b):
    '''Bson binary to numpy array'''
    return pickle.loads(b.decode())


class Comparison(Document):
    wpsA = BinaryField(required=True)
    wpsB = BinaryField(required=True)
    # paths to the videos for each
    pathA = StringField(required=True)
    pathB = StringField(required=True)
    # 0 for A, 1 for B, -1 for undecided
    label = IntField(required=False)