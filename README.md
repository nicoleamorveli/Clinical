# Clinical data analysis

This code cleans the data receive from clinical studies. Depending on the hypothesis of the study, the statistical test are chosen. 
In this example code, a new drug was tested. The sample was divided in two where group A was the control group and group b was the tested group.
Different variables (clinical variables) were measured after applying the drug. Also, demographic data was compiled for each group. 

First, descriptive statistics analysis was performed for demographic and clinical variables. The clinical variables were divided in two: quantitative and qualitative. 

Second, a t test was performed for the quantitative variables were the null hypothesis stated that there was no significant (p<0.05) difference in the meand of both groups. 

The Mann-whitney test was performed for the qualitative variables were the null hypothesis stated that there was no significant (p<0.05) difference in the meand of both groups.

The Chi-square test was performed too for some variables if they had more the two categories.

Same tests were performed for demographic data. 

The second code is for creating a latex document. The dataframes are converted in to tabular code (latex) 

