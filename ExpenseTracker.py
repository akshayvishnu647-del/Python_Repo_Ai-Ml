#1.flask-creates web page 
# 2.render_template-display html pages 
# 3.request-reads data send by user
# 4.redirect-sends the user another URL
from flask import Flask,render_template,request,redirect

#flask_sqlalchemy=works with the database and 
# ORM(obejct relational mapper)=helps to interact database using python instead of sql query
#integrates conveniently with flask
from flask_sqlalchemy import SQLAlchemy


app=Flask(__name__)#flask is the class and__name__tells flask which python module is running

#config=stores config setting for your flask application
#SQLAlchemy_DATABASE_URI=tells were database is located.
#sqlite:/// specify that you are using sqlite with relative file path
#database.db=database file name
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///database.db'
#TRACK_modification=controls sqlalchemy modification tacking feature
#False=disable the feature
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False

db=SQLAlchemy(app)#it connects flask with sqlalchemy
class Expense(db.Model):#db.model tells sqlalchemy that this class represents database table
    #It represent each individual columns 
    id=db.Column(db.Integer,primary_key=True)
    amount=db.Column(db.Float,nullable=False)
    category=db.Column(db.String(50),nullable=False)
    description=db.Column(db.String(260),nullable=True)
    date=db.Column(db.String(10),nullable=False)


@app.route('/')#'/' means root URL
def index():
    #retrieves all expense records from your expense table
    expenses=Expense.query.all()
    #tells flask to render index.html
    #render=converting a template into page that display browser
    #template are predefined structures that you can reuse and fill with different data
    return render_template('index.html',expenses=expenses)


#it tells flask which URL to use for adding and expense and which types of http request url accept
#allow the route to handle two types of http request
#'/add-the URL path for your add expense page.
@app.route('/add',methods=['GET','POST'])
def add_expense():
    if request.method =='POST':#checks whether the user submitted expense form
        #This reads the submitted data
        amount=request.form['amount']
        category=request.form['category']
        description=request.form['description']
        date=request.form['date']
        #create an expense object
        new_expense=Expense(amount=amount,category=category,description=description,date=date)
        #save the object
        db.session.add(new_expense)
        db.session.commit()
        #redirect to homepage
        return redirect('/')
        #display the form
    return render_template('add_expense.html')
#this create the database tables and starting flask web server
if __name__=='__main__':#this check whether you run the python file or not
    with app.app_context():#gives flask the app context needed by certain flask extension
        db.create_all()#create database table
    app.run(debug=True)#starts flask development server so i can open website in browser
