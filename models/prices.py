from app import db

class PriceModel(db.Model):
    __tablename__ = 'prices'

    id = db.Column(db.Integer, primary_key=True)
    company = db.Column(db.String(), nullable=False)
    date = db.Column(db.Date(), nullable=False)
    openPrice=db.Column(db.Numeric())
    closePrice=db.Column(db.Numeric())
    highPrice=db.Column(db.Numeric())
    lowPrice=db.Column(db.Numeric())
    volume=db.Column(db.Numeric())

    def __init__(self, company, date, openPrice, closePrice, highPrice, lowPrice, volume):
        self.closePrice=closePrice
        self.company=company
        self.date=date
        self.lowPrice=lowPrice
        self.highPrice=highPrice
        self.openPrice=openPrice
        self.volume=volume

    def __repr__(self) -> str:
        return f'<Price for {self.company} for day {self.date} is {self.closePrice}>'