CPS.csv

The first row displays the attributes of the CPS data.
Each data record contains:
index: integer
tax: integer
income: integer
csp: integer
age: integer
educ: integer
marital: [1,7]
race: [1,2]
sex: [1,2]
ss: integer

Number of record: 49436

One suggested model:
log(income) ~ age, educ, marital, sex


