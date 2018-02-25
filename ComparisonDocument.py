from mongoengine import (
    Document, StringField, IntField,
    BinaryField, BooleanField
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


class ComparisonQueue:
    def __init__(self, allowed_labels=[0,1]):
        self.allowed_labels = allowed_labels


    @property
    def queue(self):
        return Comparison.objects(label__in=self.allowed_labels)


    def sample(self, num=1):
        q = self.queue
        size = q.count()
        if num == 1:
            idx = np.random.choice(q.count())
            return q[idx]
        else:
            idcs = np.random.choice(q.count(), size=num, replace=False)
            return [self.q[idx] for idx in idcs]


    def __len__(self):
        return self.queue.count()
