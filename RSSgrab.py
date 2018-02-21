from grab import Grab
from RSSdb import Cocktails, Connector
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy import Sequence


class Grabber:
    def __init__(self):
        self.g = Grab()
        self.db_instance = None

    def connect_db(self):
        self.db_instance = Connector()

    def run(self):
        if not self.db_instance:
            print('DB is not connected')
        for page in range(0, 10000):
            self.g.go('https://www.webtender.com/db/drink/{}'.format(page))
            if self.g.doc.code != 404:
                tables = self.g.doc.select('//table')
                take_id = None
                for idx, table in enumerate(tables):
                    if 'Ingredients:' in table.text():
                        take_id = idx
                        break

                if take_id is not None:
                    name = tables[take_id].select('//h1').text()
                    ing = tables[take_id].select('//ul').text()
                    ing_html = tables[take_id].select('//ul').html()
                    mix = tables[take_id].select('//td/p').text()

                    to = Cocktails(name=name, ing=ing, ing_html=ing_html, mix=mix)
                    print('Try to write page: {}'.format(page))
                    if self.db_instance:
                        self.db_instance.db_writer(to)
                else:
                    print('Page is empty: {}'.format(page))

            else:
                print('Page not found: {}'.format(page))

            if self.db_instance and (page % 10 == 0):
                self.db_instance.commit()
        if self.db_instance:
            self.db_instance.commit()


if __name__ == '__main__':
    g = Grabber()
    g.connect_db()
    g.run()