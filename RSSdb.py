from grab import Grab
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy import Sequence
import pandas as pd

Base = declarative_base()


class Cocktails(Base):
     __tablename__ = 'cocktails'

     id = Column(Integer, Sequence('cocktail_id_seq'), primary_key=True)
     name = Column(String(250))
     ing = Column(String)
     ing_html = Column(String)
     mix = Column(String)


class Connector:
    def __init__(self):
        self.engine = create_engine('sqlite:///bartender.db', echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def get_values(self, name):
        return pd.read_sql_table(name, self.engine)

    def create_table(self, dataframe,  name):
        return dataframe.to_sql(name, self.engine, index=False, if_exists='replace')

    def db_writer(self, data):
        if self.check_data(data):
            self.session.add(data)
        else:
            print('Cant add wrong format or duplicated')

    def check_data(self, data):
        return data.name not in list(map(lambda x: x.name, self.session.query(Cocktails)))

    def commit(self):
        self.session.commit()
        print('Committed')

    # def __del__(self):
    #     self.session.close()