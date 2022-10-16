# App to render dashboards for feature importances using an uploaded dataframe.

Program flow will work as follows:
1. User uploades dataframe to the application;
2. User chooses target column;
3. User chooses the predictor columns (from existing DataFrame columns);
4. User chooses the metric to be used (Gini Index vs. Odds i.e. Decision Tree vs Logistic Regression);
5. User gets a table as output which shows feature importances with option to export the table to a csv.

**Notes**: Following are not implemented, and may either be implemented in the future or may never be implemented:
1. Works with a classification problems;
2. Works with a binary class of target variable only; 

---
## Handle file uploads
- [ ] Handle uploads;
- [ ] User can choose to upload a csv or excel file;
- [ ] User will choose row containing headers (-1 if no headers);
- [ ] If excel, user will be able to choose a sheet;
- [ ] If csv, user will able to choose delimiter as well.
- [ ] User will get a preview of the first 5 rows of the data;