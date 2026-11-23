"""Compatibility fixes for the installed AppWorld / SQLAlchemy combination."""
import os


def install_database_url_fix():
    """Preserve the canonical in-memory DB identity across start/end reads.

    SQLAlchemy renders `file:` as `file%3A`. AppWorld 0.2.0.dev0 parses that
    display string as an actual DB name and opens an empty second database.
    The URL object's database field retains the actual name.
    """
    from appworld.apps.lib.models.db import Database
    original=Database._get_engine_to_details
    if getattr(original,'jev_database_url_fix',False):
        return

    def details(engine):
        url,path,home,storage,connection=original(engine)
        if engine.url.query.get('mode')=='memory':
            name=engine.url.database
            if not isinstance(name,str) or not name.startswith('file:'):
                raise ValueError('Expected a named SQLite memory database')
            home=os.path.dirname(':memory:'+name.removeprefix('file:'))
        return url,path,home,storage,connection

    details.jev_database_url_fix=True
    Database._get_engine_to_details=staticmethod(details)


def evaluate_world(world):
    """Do not silently turn a broken SQLite connection into an agent failure."""
    from appworld.evaluator import TestTracker
    from sqlalchemy.exc import SQLAlchemyError
    original=TestTracker.__exit__
    def record(tracker,kind,exception,tb):
        underlying=getattr(exception,'exception',exception)
        if isinstance(underlying,(SQLAlchemyError,FileNotFoundError,ImportError)):
            raise underlying
        return original(tracker,kind,exception,tb)
    TestTracker.__exit__=record
    try:
        return world.evaluate()
    finally:
        TestTracker.__exit__=original
