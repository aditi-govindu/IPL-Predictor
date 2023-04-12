# IPL-Predictor
Based on historical data predict total runs scored by the batting team, at the end of 6 overs for IPL 2021.

# Problem statement
Given certain input parameters regarding an innings of a T20 cricket match, predict the total runs scored by the batting team at the end of 6 overs.

*Input data*

We are providing the link to the Dataset (Source: cricsheet.org) which contains historic data of T20 matches that have occurred in the past. This dataset may be used by the candidate teams to train an ML model or come up with a data analytics based algorithm that can perform the required prediction. In addition to this, I have used the **ESPN Cric Info** site to collect batsmen strike rate and bowler economy.

*Output data*

The following will be provided as input test case data: *venue, innings, batting_team, bowling_team, batsmen who batted during the first 6 overs, bowlers who bowled during the first 6 overs.*
The candidates may consume the data and preprocess or convert them in any manner for making it work as per their developed model.
**Points scored by each team = R2 error value (sum of square of error) between the actual score and predicted score (The lower the points, better is the prediction outcome).**

*Permitted packages*

Following is a list of python packages that can be imported and used in **predictor.py**, as teams see fit.
* h5py==2.10.0
* jupyter==1.0.0
* Keras==2.4.3
* Keras-Preprocessing==1.1.2
* numpy==1.19.5
* pandas==1.2.4
* scikit-learn==0.24.1
* scipy==1.6.2
* seaborn==0.11.1
* tensorflow>=2.11.1
* torch==1.8.1

# References
* https://internalapp.nptel.ac.in/contest/contest_details.html
* https://stats.espncricinfo.com/ci/engine/records/batting/most_runs_career.html?id=117;type=trophy
* https://stats.espncricinfo.com/ci/engine/records/bowling/most_wickets_career.html?id=117;type=trophy
