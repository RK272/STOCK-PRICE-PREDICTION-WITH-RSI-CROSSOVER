This is the code for stock price prediction using rsi crossover 
when rsi(14) crosess 40 line and it want to predict if stock rsi reach 60 without failure then stock price also increase

DATA REQUIRED stock data datetime ,open,high,low,close
and date time given in format "%d-%m-%Y %H:%M" if u use any other format u need to change inside datatransformation.py in src/component /datatransformation line no 133

used python 3.7 if use any other version some dependency issue will come

database used casandra
so u need to connect with your database by changing secure-connect-test.zip and test.zip in src/casandrafiles folder
and you need to pass key name and table name of casandra database in src/constant/database.py


step 
 1 pip install -r requirements.txt
 2-change database token and bundle and keyspacename,table name 
 3 if your datetime column is this formate then change . %Y-%m-%d %H:%M:%S format instruction given above
 4 then in terminal just type python main.py