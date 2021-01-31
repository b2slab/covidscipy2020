import sqlite3

import click
from flask import current_app, g
from flask.cli import with_appcontext

from /tmp/covidscipy2020/PythonApi import getEntries 

def data_adquisition(pat_names, data_values):
    for val in data_values:
        data_plot={val:[]}
    for pat in pat_names:
        pat_values=getEntries.getSpecific(pat_names,data_values)
        for i in range(len(pat_values)):
            data_plot={data_values(i):dataplot[data_values(i)].append(pat_values(i))}
    return data_plot


################
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db


def close_db(e=None):
    db = g.pop('db', None)

    if db is not None:
        db.close()

##################


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

##################

def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
