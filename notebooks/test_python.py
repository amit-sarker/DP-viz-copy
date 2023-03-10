#!/usr/bin/env python
# coding: utf-8

# In[128]:


import sys

import ipywidgets as widgets

sys.path.append('../')
sys.path.append('../src')

import src.backend
import src.workload_builder as builder
from mbi import Domain
import src.plots
import altair as alt

alt.renderers.enable('default')
from IPython.display import display, clear_output

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

alt.data_transformers.disable_max_rows()

# In[129]:


data_path = '../data/CPS/CPS.csv'
cps_domain = Domain(attrs=('age', 'income', 'marital', 'race'), shape=(100, 100, 7, 4))
questions = ["Warm Up: Type 'Yes' if you are ready to begin!",
             "Question 1/6: How many Chinese and American Indians have 'Widowed' as their marital status?",
             "Question 2/6: How many people over the age of 65 have income between 100,000 and 149,000?",
             "Question 3/6: What is the second most common marital status of people with income in the 100,000 to 199,000 range?",
             "Question 4/6: Choose the top 3 age groups that have marital status: 'Never Married'?",
             "Question 5/6: In the 100,000 to 199,000 income range, rank the top 3 races in terms of count from greatest to least.",
             "Question 6/6: What is the second most common race in the 45-54 age group?"]

marital_race = {'marital': 1, 'race': 1}
age_income = {'age': 10, 'income': 10}
marital_income = {'income': 10, 'marital': 1}
age_marital = {'age': 10, 'marital': 1}
race_income = {'income': 10, 'race': 1}
age_race = {'age': 10, 'race': 1}

visualizations = [None, marital_race, age_income, marital_income, age_marital, race_income, age_race]
prev_spec = None
curr_spec = None
epsilon_increments = [None, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
budget = 1.0
max_tries = 10
global index
index = 0
tries = 0
back_end = None

# In[130]:


cps_domain

# # Logging

# In[131]:


import logging


class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {
            'width': '100%',
            'height': '160px',
            'border': '1px solid black'
        }
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record + '\n'
        }
        self.out.outputs = (new_output,) + self.out.outputs

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()


logger = logging.getLogger(__name__)
handler = OutputWidgetHandler()
handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# In[132]:


plot_output = widgets.Output()

# In[133]:


# plot_output


#  ## Dropdown Boxes

# In[134]:


columns = ['Age', 'Income', 'Marital Status']
translate = {'Income': 'income', 'Marital Status': 'marital', 'Age': 'age', 'Race': 'race'}

# ## Buttons

# In[135]:


import numpy as np

submit_btn = widgets.Button(description='Submit')
make_it_better = widgets.Button(description='Remeasure')


def on_click_submit(obj):
    global index, survey_answers, visualizations, epsilon_increments, curr_spec, back_end
    logger.info('Clicked submit')
    if answer.value == '':
        return
    val = answer.value
    answer.value = answer.placeholder
    survey_answers[index] = val

    if index + 1 < len(questions):
        index += 1
        prompt.value = questions[index]
    else:
        with plot_output:
            clear_output()
        submit_btn.close()
        interface.close()
        bar_label.close()
        prompt.value = 'Tutorial finished, thanks for your participation!'
        display(prompt)
        return

    back_end = src.backend.initialize_backend(cps_domain, data_path, budget=1.0)

    hist = builder.histogram_workload(cps_domain.config, bin_widths=visualizations[index])

    # logger.info(hist.matrix.matrix)

    y_hat, strategy_matrix = back_end.measure_hdmm(workload=hist, eps=epsilon_increments[index],
                                                   restarts=20)  # got y_hat

    # B = np.linalg.pinv(strategy_matrix)

    # logger.info("printing y_hat")
    # logger.info(B)
    logger.info("printing strategy matrix")
    #     logger.info(strategy_matrix.matrix)

    column_names = list(visualizations[index].keys())
    curr_spec = back_end.display(hist)

    with plot_output:
        plot_output.clear_output()
        plot = src.plots.linked_hist(column_names[0], column_names[1], data=curr_spec.reset_index(column_names),
                                     display_true=False, history=False)
        logger.info(plot)
        #         display("Updating...")
        display(plot)


#         altair_viewer.show(plot)

def on_click_make_it_better(obj):
    global index, visualizations, tries
    logger.info('Clicked remeasure, question: {}'.format(index))

    key_val = [x for x in visualizations[index].items()]
    logger.info(key_val)
    measure_dict = {'left': key_val[0], 'right': key_val[1]}
    binning(measure_dict, epsilon=epsilon_increments[index])


submit_btn.on_click(on_click_submit)
make_it_better.on_click(on_click_make_it_better)

# In[ ]:


# In[136]:


# import numpy as np
#
# submit_btn = widgets.Button(description='Submit')
# make_it_better = widgets.Button(description='Remeasure')
# start_btn = widgets.Button(description='Start')
#
# # def on_click_start():
#
#
# def on_click_submit(obj):
#     global index, survey_answers, visualizations, epsilon_increments, curr_spec, back_end
#     logger.info('Clicked submit')
#     if answer.value == '':
#         return
#     val = answer.value
#     answer.value = answer.placeholder
#     survey_answers[index] = val
#
#     if index+1 < len(questions):
#         index += 1
#         prompt.value = questions[index]
#     else:
#         with plot_output:
#             clear_output()
#         submit_btn.close()
#         interface.close()
#         bar_label.close()
#         prompt.value = 'Tutorial finished, thanks for your participation!'
#         display(prompt)
#         return
#
#     back_end = src.backend.initialize_backend(cps_domain, data_path, budget=1.0)
#
#     hist = builder.histogram_workload(cps_domain.config, bin_widths=visualizations[index])
#
#     # logger.info(hist.matrix.matrix)
#
#     y_hat, strategy_matrix = back_end.measure_hdmm(workload=hist, eps=epsilon_increments[index], restarts=20) # got y_hat
#
#     # B = np.linalg.pinv(strategy_matrix)
#
#     # logger.info("printing y_hat")
#     # logger.info(B)
#     logger.info("printing strategy matrix")
# #     logger.info(strategy_matrix.matrix)
#
#     column_names = list(visualizations[index].keys())
#     curr_spec = back_end.display(hist)
#
#     with plot_output:
#         plot_output.clear_output()
#         plot = src.plots.linked_hist(column_names[0], column_names[1], data=curr_spec.reset_index(column_names), display_true=False, history=False)
#         logger.info(plot)
# #         display("Updating...")
#         display(plot)
# #         altair_viewer.show(plot)
#
# def on_click_make_it_better(obj):
#     global index, visualizations, tries
#     logger.info('Clicked remeasure, question: {}'.format(index))
#
#     key_val = [x for x in visualizations[index].items()]
#     logger.info(key_val)
#     measure_dict = {'left': key_val[0], 'right': key_val[1]}
#     binning(measure_dict, epsilon = epsilon_increments[index])
#
# submit_btn.on_click(on_click_submit)
# make_it_better.on_click(on_click_make_it_better)


# ## Text Boxes

# In[137]:


# global index
prompt = widgets.Textarea(
    value=questions[0],
    placeholder='Type',
    description='',
    disabled=True,
    layout=widgets.Layout(width='1000px', height='30px')
)

answer = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

answer2 = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

answer3 = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

answer4 = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

answer5 = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

answer6 = widgets.Textarea(
    value='',
    placeholder='',
    description='Answer:',
    layout=widgets.Layout(width='300px', height='40px')
)

# In[138]:


text_area = []
for i in range(len(questions)):
    text_area.append(answer)

# In[139]:


back_end = src.backend.initialize_backend(cps_domain, data_path, budget=budget)

progress_bar = widgets.FloatProgress(min=0.0, max=max_tries)  # instantiate the bar
progress_bar.style.bar_color = 'red'
progress_bar.description = str(tries) + '/' + str(max_tries)
budget_spent = widgets.Label(value='Tries')
num_answers = len(questions)
survey_answers = [0] * num_answers


def binning(measure_dict, epsilon=None, group_income=None):
    global tries, curr_spec, back_end

    left_col = measure_dict['left']
    right_col = measure_dict['right']

    widths = {left_col[0]: left_col[1], right_col[0]: right_col[1]}

    hist = builder.histogram_workload(cps_domain.config, bin_widths=widths)

    if epsilon is not None:
        if tries + 1 >= max_tries:
            make_it_better.close()
        if tries + 1 > max_tries:
            return
        tries += 1
        progress_bar.value = tries
        progress_bar.description = str(tries) + '/' + str(max_tries)
        y_hat, strategy_matrix = back_end.measure_hdmm(workload=hist, eps=epsilon, restarts=20)

    prev_spec = curr_spec
    prev_spec.rename(columns={'error': 'error_prev', 'plus_error': 'plus_error_prev', 'minus_error': 'minus_error_prev',
                              'true_count': 'true_count_prev', 'noisy_count': 'noisy_count_prev'}, inplace=True)
    curr_spec = back_end.display(hist)  ################# Cache_search function called

    spec = curr_spec.join(prev_spec, on=[left_col[0], right_col[0]]).reset_index([left_col[0], right_col[0]])
    spec = spec.round(0)

    print(spec['plus_error'])

    with plot_output:
        plot_output.clear_output()
        plot = src.plots.linked_hist_test(left_col[0], right_col[0], data=spec, projection=True, label=False)
        display(plot)


box_layout = widgets.Layout(display='flex',
                            flex_flow='column',
                            align_items='flex-start',
                            color='black',
                            width='80%')

bar_label = widgets.HBox([budget_spent, progress_bar, make_it_better])
prompt_answer = widgets.VBox([answer, submit_btn], layout=box_layout)

# In[140]:


display(bar_label)

interface = widgets.VBox([prompt, plot_output, prompt_answer])
display(interface)

# In[ ]:


# In[141]:


print(handler.show_logs())

# In[142]:


# print(questions)


# In[ ]:


# In[ ]:



