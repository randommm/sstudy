from peewee import *
import os

try:
    pgdb = os.environ['pgdb']
    pguser = os.environ['pguser']
    pgpass = os.environ['pgpass']
    pghost = os.environ['pghost']
except KeyError:
    try:
        sqlitefilename = os.environ['sqlitefilename']
    except KeyError:
        sqlitefilename = 'results.sqlite3'
    db = SqliteDatabase(sqlitefilename)
else:
    if 'pgport' in os.environ:
        pgport = os.environ['pgport']
    else:
        pgport = 5432

    db = PostgresqlDatabase(pgdb, user=pguser, password=pgpass,
    host=pghost, port=pgport)

class Result(Model):
    # Data settings
    hidden_size = IntegerField()
    seed = IntegerField()
    dropout = BooleanField()
    batch_normalization = BooleanField()

    # Results
    mse_train = DoubleField()
    mse_test = DoubleField()

    class Meta:
        database = db

Result.create_table()
