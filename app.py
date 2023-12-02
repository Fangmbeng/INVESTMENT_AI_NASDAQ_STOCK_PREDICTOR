import psycopg2
from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

#initialize app
app = Flask(__name__)
#Configure the app using the Config Class
app.config['SECRET_KEY']='my-name'
app.config['SQLALCHEMY_DATABASE_URI']='postgresql://jvnnhxji:bApgkzBgjCKvK5VzSZVyNL6AukDinlUb@berry.db.elephantsql.com/jvnnhxji'
app.config['SQLQLCHEMY_TRACK_MODIFICATIONS']=False

#initializes a db ORM for the app
db = SQLAlchemy(app)

#import PriceModel and attributes for migration to database
from models.prices import PriceModel
from prices.prices import getPrices
migrate = Migrate(app, db)

if __name__ =='__main__':
    app.run(debug=True)