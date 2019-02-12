#test file



'''
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp

import pandas

df = pandas.read_csv("data_columns.csv")

# Fits the model with the interaction term
# This will also automatically include the main effects for each factor
model = ols('Yield ~ C(Fert)*C(Money)', df).fit()

# Seeing if the overall model is significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

print("\n\n")

# Reviewing the model summary
print(model.summary())

print("\n\n")

#Seeing the anova statistics for the independent variables (stimulus_type and individual_type)
print(sm.stats.anova_lm(model, typ= 2))
'''


import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pg


# Let's assume that we have a balanced design with 30 students in each group
n = 30
months = ['August', 'January', 'June']

# Generate random data
np.random.seed(1234)
control = np.random.normal(5.5, size=len(months) * n)
meditation = np.r_[ np.random.normal(5.4, size=n),
                    np.random.normal(5.8, size=n),
                    np.random.normal(6.4, size=n) ]

# Create a dataframe
df = pd.DataFrame({'Scores': np.r_[control, meditation],
                   'Time': np.r_[np.repeat(months, n), np.repeat(months, n)],
                   'Group': np.repeat(['Control', 'Meditation'], len(months) * n),
                   'Subject': np.r_[np.tile(np.arange(n), 3),
                                    np.tile(np.arange(n, n + n), 3)]})


print(df)


sns.set()
sns.pointplot(data=df, x='Time', y='Scores', hue='Group', dodge=True, markers=['o', 's'],
	      capsize=.1, errwidth=1, palette='colorblind')


'''
df.groupby(['Time', 'Group'])['Scores'].agg(['mean', 'std']).round(2)

# Compute the two-way mixed-design ANOVA
aov = pg.mixed_anova(dv='Scores', within='Time', between='Group', data=df)
# Pretty printing of ANOVA summary
pg.print_table(aov)


posthocs = pg.pairwise_ttests(dv='Scores', within='Time', between='Group', subject='Subject',
			      effects='interaction', data=df)
pg.print_table(posthocs)
'''