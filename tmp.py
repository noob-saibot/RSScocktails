from grab import Grab
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy import Sequence
from preproc import Cocktails
engine = create_engine('sqlite:///bartender.db', echo=False)

Session = sessionmaker(bind=engine)
session = Session()

print(len(list(session.query(Cocktails))))
for instance in session.query(Cocktails):
    print(instance.name)
    print(instance.mix)

session.close()