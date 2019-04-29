#All the statistical functions are defined here
#NTBD: try to create a function that does statistical anlaysis only on a subsection of the data (eg: only textual stimuli etc)

import pingouin as pg


#Mixed anova
def mixed_anova_calculation(meta, data, subject_parameters, stimuli_parameters):
	"""Function that calls the mixed anova function

	Parameters
	----------

	meta: string
		The parameter that is being considered for anova analysis
	data: pandas dataframe
		The dataframe that contains the values for ANOVA analysis
	subject_parameters: list of strings
		The list of between subject factors
	stimuli_parameters: list of strings
		The list of within subject factors
	"""

	
	aov = pg.mixed_anova(dv=meta, within=stimuli_parameters[0], between=subject_parameters[0], subject = 'subject', data=data)
	pg.print_table(aov)

	posthocs = pg.pairwise_ttests(dv=meta, within=stimuli_parameters[0], between=subject_parameters[0], subject='subject', data=data)
	pg.print_table(posthocs)
	
#Repeated measures anova
def rm_anova_calculation(meta, data, stimuli_parameters):
	"""Function that calls the repeated measures anova function

	Parameters
	----------

	meta: string
		The parameter that is being considered for anova analysis
	data: pandas dataframe
		The dataframe that contains the values for ANOVA analysis
	stimuli_parameters: list of strings
		The list of within subject factors
	"""

	aov = pg.rm_anova(dv=meta, within= stimuli_parameters, subject = 'subject', data=innocent_data)
	pg.print_table(aov)

#T test 
def ttest_calculation(meta, data, stimuli_parameters, subject_parameters):
	"""Function that calls the t-test function

	Parameters
	----------

	meta: string
		The parameter that is being considered for anova analysis
	data: pandas dataframe
		The dataframe that contains the values for ANOVA analysis
	stimuli_parameters: list of strings
		The list of within subject factors
	subject_parameters: list of strings
		The list of between subject factors
	"""

	posthocs = pg.pairwise_ttests(dv=meta, within= stimuli_parameters, between= subject_parameters, subject='subject', data=data)
	pg.print_table(aov)