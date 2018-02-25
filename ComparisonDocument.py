from mongoengine import (
    Document, StringField, IntField,
    BinaryField
)


class Comparison(Document):
    wpsA = BinaryField(required=True)
    wpsB = BinaryField(required=True)
    # paths to the videos for each
    pathA = StringField(required=True)
    pathB = StringField(required=True)
    # 0 for A, 1 for B, -1 for undecided
    label = IntField(required=False)