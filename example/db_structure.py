from peewee import *
import os

db = SqliteDatabase('results.sqlite3')

class Result(Model):
    # Data settings
    data_distribution = TextField()
    method = TextField()
    no_instances = DoubleField()

    # Results
    score = DoubleField()
    elapsed_time = DoubleField()

    class Meta:
        database = db

Result.create_table()
