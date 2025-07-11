I DID NOT UPLOAD THE CSV FILES WHERE I HAVE THE MARKET MINUTE DATA BECAUSE THEY ARE HUGE.
In this project I am developing a backtesting platform with minute data. On the CROSS_SECTIONAL_ALPHAS IN-SAMPLE.py, I generate trading signals with daily candles based on the strategy (I save the strategies on the STRATEGIES folder), and then I put those signals on a minute by minute dataframe. Then I backtest these daily signals with the minute dataframe and I simulate slippage, fees, stop loss, take profit and capital allocation. Here I am trying to find efficient portafolio stratgies that allocate capital to different assets.
My goal is to have a platform that lets me backtest hypostheis or trading strategies accross many assets that will save me time. 

I EXPLORE USING DIFFERENT PORTAFOLIO STRATGEIES, AMONG THEM THE KELLY CRITERION WHICH OPTIMIZES THE WEIGHTS IN THE PORTAFOLIO TO MAXIMIZE PORTAFOLIO RETURNS..

RESULTS TESTING THE KELLY CRITERION :
<img width="3825" height="1502" alt="Captura de pantalla 2025-07-10 172715" src="https://github.com/user-attachments/assets/1bcbc012-0d54-4141-9251-5a3f8518bc4b" />
THIS IS THE TRADE BOOK: 
<img width="3777" height="750" alt="Captura de pantalla 2025-07-10 173654" src="https://github.com/user-attachments/assets/ee3bd599-1865-4891-a9d3-160cb3049758" />



