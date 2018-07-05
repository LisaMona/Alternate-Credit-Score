# Alternate-Credit-Score

This work involves calculating the Alternate Credit Score of the individuals based on their non-financial datapoints.
Unlike the Traditional Credit Score calculation, factors like card amount, loan amout, income etc are not considered.

Assuming we don't already have the Traditional Credit Score of persons, a dataset for 600 individuals has been prepared taking the following factors into account.
1- Mobile Bill Payment
2- Elecricity Bill Payment
3- Qualification
4- Occupation
5- Location
These are available in the "Dataset" folder.


For each of the above mentioned measures, a score was calculated for each of them and are store in "Dataset_with_scores folder".
Then giving some meaningful weight for each of the scores calculated, average of all the scores was found.
The credit scores were normalized between 0-100, The higher the score, more trustworthy the persion is.
Now appending all these data features along with their credit scores(avg), the Alternate Credit Score was calculated in "model.py".
Data Preprocessing was carried out and a Multiple linear regression model was fit to the data-points that could calculate the Scores with an accuracy of 81%.
