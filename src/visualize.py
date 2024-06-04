import ast
import plotly.graph_objects
import inspectify
import math
import os
import pandas
import sqlite3
import subprocess

def sync_data(source, destination):
  cmd = ["rsync", "-avz", "--progress", source, destination]

  try:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print("Sync completed successfully.")
    print(result.stdout)
  except subprocess.CalledProcessError as e:
    print("Error in sync:")
    print(e.stderr)

def load_config_data(db_path):
  loaded_config = {}
  try:
    with sqlite3.connect(db_path) as conn:
      cursor = conn.cursor()
      cursor.execute("SELECT key, value FROM CONFIG")
      rows = cursor.fetchall()

    # Process each row to infer the type and construct a nested dictionary
    for key, value in rows:
      # Infer the type
      if value.isdigit():
        parsed_value = int(value)
      elif all(char.isdigit() or char == '.' for char in value):
        try:
          parsed_value = float(value)
        except ValueError:
          parsed_value = value
      elif value.startswith('{') and value.endswith('}'):
        try:
          # Attempt to parse as a dictionary
          parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
          # If parsing fails, keep the original value
          parsed_value = value
      else:
        parsed_value = value

      # Create nested dictionaries based on key structure
      key_parts = key.split('__')
      d = loaded_config
      for part in key_parts[:-1]:
        if part not in d:
          d[part] = {}
        d = d[part]
      d[key_parts[-1]] = parsed_value
  except sqlite3.Error:
    # If the CONFIG table doesn't exist or any other SQL error occurs
    loaded_config = {}

  return loaded_config

def load_configs_from_folder(folder_path):
  configs = []
  for file in os.listdir(folder_path):
    if file.endswith(".db"):
      db_path = os.path.join(folder_path, file)
      config = load_config_data(db_path)
      config['db_path'] = db_path
      configs.append(config)
  return configs

def format_value_for_expression(value):
  if isinstance(value, str):
    # Add quotes around strings
    return f"\"{value}\""
  return value

def match_config_with_filter(config, filter_expr):
  for key, value in filter_expr.items():
    if isinstance(value, dict):
      # Recursive call for nested dictionaries
      if key not in config or not match_config_with_filter(config[key], value):
        return False
    elif isinstance(value, list):
      # Handle lists of expressions
      for i, expr in enumerate(value):
        if i >= len(config.get(key, [])):
          return False
        config_value = format_value_for_expression(config[key][i])
        expression = expr.replace("{}", str(config_value))
        condition = eval(expression)
        assert type(condition) == bool, f"Filter {key}:{expression} must express a bool."
        if not condition:
          return False
    else:
      # Handle individual expressions
      config_value = format_value_for_expression(config.get(key))
      expression = value.replace("{}", str(config_value))
      condition = eval(expression)
      assert type(condition) == bool, f"Filter {key}:{expression} must express a bool."
      if not condition:
        return False
  return True

def filter_configs(configs, filter_expr):
  matching_db_paths = []
  for config in configs:
    if match_config_with_filter(config, filter_expr):
      matching_db_paths.append(config.get('db_path'))
  return matching_db_paths

def load_policy_performance_data(db_path, xaxis_choice, yaxis_choice):
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Execute SQL query based on x-axis choice
  cursor.execute(f'SELECT DISTINCT policy_id, {xaxis_choice} FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
  training_data = cursor.fetchall()

  try:
    cursor.execute('SELECT policy_id, num_training_episodes, num_total_function_evaluations, num_total_timesteps FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
    rows = cursor.fetchall()
    policy_id_to_x_values = {policy_id: {column_name: value for column_name, value in zip(['num_training_episodes', 'num_total_function_evaluations', 'num_total_timesteps'], row)}
                 for policy_id, *row in rows}
  except sqlite3.OperationalError:
    cursor.execute('SELECT policy_id, num_training_episodes FROM CONSTRUCTED_POLICIES WHERE policy_id >= 1')
    rows = cursor.fetchall()
    policy_id_to_x_values = {policy_id: {'num_training_episodes': num_training_episodes}
                 for policy_id, num_training_episodes in rows}

  avg_function_evaluations, std_dev_evaluations = [], []
  for policy_id, _ in training_data:
    cursor.execute(f'SELECT {yaxis_choice} FROM EVALUATION_EPISODES WHERE policy_id = ?', (policy_id,))
    evaluations = [e[0] for e in cursor.fetchall()]
    assert evaluations[0] is not None, f"The database {db_path} has a table EVALUATION_EPISODES with a cell of column {yaxis_choice} that is NULL!"
    num_evaluation_episodes = len(evaluations)  # Assuming number of episodes is length of evaluations

    avg_evaluations = sum(evaluations) / len(evaluations) if evaluations else None
    std_dev = math.sqrt(sum((e - avg_evaluations) ** 2 for e in evaluations) / len(evaluations)) if evaluations else 0
    avg_function_evaluations.append(avg_evaluations if avg_evaluations is not None else 0)
    std_dev_evaluations.append(std_dev)

  policy_ids, num_training_timesteps_or_num_training_fes = zip(*training_data) if training_data else ([], [])

  cursor.execute(f'SELECT {yaxis_choice} FROM EVALUATION_EPISODES WHERE policy_id = -1')
  baseline_evaluations = [e[0] for e in cursor.fetchall()]
  baseline_avg_length = sum(baseline_evaluations) / len(baseline_evaluations)
  baseline_variance = sum((e - baseline_avg_length) ** 2 for e in baseline_evaluations) / (len(baseline_evaluations) - 1) if len(baseline_evaluations) > 1 else 0
  baseline_std_dev = math.sqrt(baseline_variance)

  baseline_upper_bound = [baseline_avg_length + baseline_std_dev] * len(num_training_timesteps_or_num_training_fes)
  baseline_lower_bound = [baseline_avg_length - baseline_std_dev] * len(num_training_timesteps_or_num_training_fes)

  data = [
    plotly.graph_objects.Scatter(x=num_training_timesteps_or_num_training_fes, y=avg_function_evaluations, mode='lines+markers', name='#FEs until optimum', line=dict(color='blue', width=4)),
    plotly.graph_objects.Scatter(x=num_training_timesteps_or_num_training_fes, y=[avg + std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', line=dict(color='rgba(173,216,230,0.2)'), name='Upper Bound (Mean + Std. Dev.)'),
    plotly.graph_objects.Scatter(x=num_training_timesteps_or_num_training_fes, y=[avg - std for avg, std in zip(avg_function_evaluations, std_dev_evaluations)], mode='lines', fill='tonexty', line=dict(color='rgba(173,216,230,0.2)'), name='Lower Bound (Mean - Std. Dev.)'),
    plotly.graph_objects.Scatter(x=[min(num_training_timesteps_or_num_training_fes), max(num_training_timesteps_or_num_training_fes)] if num_training_timesteps_or_num_training_fes else [0], y=[baseline_avg_length, baseline_avg_length], mode='lines', name='Theory: √(𝑛/(𝑛 − 𝑓(𝑥)))', line=dict(color='orange', width=2, dash='dash')),
    plotly.graph_objects.Scatter(x=num_training_timesteps_or_num_training_fes, y=baseline_upper_bound, mode='lines', line=dict(color='rgba(255, 165, 0, 0.2)'), name='Upper Bound (Baseline Variance)'),
    plotly.graph_objects.Scatter(x=num_training_timesteps_or_num_training_fes, y=baseline_lower_bound, mode='lines', fill='tonexty', line=dict(color='rgba(255, 165, 0, 0.2)'), name='Lower Bound (Baseline Variance)'),
  ]

  conn.close()
  return data

def policy_performance(db_path, xaxis_choice, yaxis_choice):
  data = load_policy_performance_data(db_path, xaxis_choice, yaxis_choice)
  # Define the layout with larger dimensions and enhanced appearance
  layout = plotly.graph_objects.Layout(
    titlefont=dict(size=24),  # Bigger title font size
    xaxis=dict(
      title=xaxis_choice.replace('_', ' ').title(),
      titlefont=dict(size=18),  # Bigger axis title font size
      tickfont=dict(size=14),  # Bigger tick labels font size
      gridcolor='lightgrey',  # Grid color
      gridwidth=2,  # Grid line width
    ),
    yaxis=dict(
      title='#FEs until optimum',
      titlefont=dict(size=18),  # Bigger axis title font size
      tickfont=dict(size=14),  # Bigger tick labels font size
      gridcolor='lightgrey',  # Grid color
      gridwidth=2,  # Grid line width
    ),
    font=dict(family='Courier New, monospace', size=18, color='RebeccaPurple'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(245, 245, 245, 1)',
    width=1100,  # Width of the figure
    height=600,  # Height of the figure
    margin=dict(l=50, r=50, b=100, t=100, pad=4),  # Margins to prevent cutoff
    showlegend=False,  # This will remove the legend
  )

  fig = plotly.graph_objects.Figure(data=data, layout=layout)
  fig.show()

def get_policy_id_for_timesteps(db_path, total_timesteps):
  try:
    with sqlite3.connect(db_path) as conn:
      cursor = conn.cursor()
      # SQL query to fetch policy_id based on the total timesteps
      cursor.execute('SELECT policy_id FROM CONSTRUCTED_POLICIES WHERE num_total_timesteps = ?', (total_timesteps,))
      result = cursor.fetchone()
      return result[0] if result else None
  except sqlite3.Error as e:
    print(f"SQLite error: {e}")
    return None

def generate_fitness_lambda_plot(db_path, policy_total_timesteps, xaxis_choice, yaxis_choice):

  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()

  # Fetch baseline fitness-lambda data (policy_id = -1)
  cursor.execute('SELECT fitness, crossover_size FROM POLICY_DETAILS WHERE policy_id = -1')
  baseline_fitness_lambda_data = cursor.fetchall()

  baseline_curve = plotly.graph_objects.Scatter(
    x=[d[0] for d in baseline_fitness_lambda_data],
    y=[d[1] for d in baseline_fitness_lambda_data],
    mode='lines+markers',
    name='Baseline Fitness-Lambda',
    line=dict(color='orange', width=4)
  )

  policy_id = get_policy_id_for_timesteps(db_path, policy_total_timesteps)
  assert policy_id is not None, f"The number of total timesteps {policy_total_timesteps:,} has resulted in a non-existing policy ID!"

  # Fetch mean and variance of initial fitness for the specified policy
  cursor.execute('SELECT mean_initial_fitness, variance_initial_fitness FROM CONSTRUCTED_POLICIES WHERE policy_id = ?', (policy_id,))
  fitness_stats = cursor.fetchone()
  mean_initial_fitness = std_dev_initial_fitness = None
  if fitness_stats and fitness_stats[0]:
    mean_initial_fitness = fitness_stats[0]
    variance_initial_fitness = fitness_stats[1]
    std_dev_initial_fitness = math.sqrt(variance_initial_fitness)

  # Fetch fitness-lambda data for the specified policy
  cursor.execute('SELECT fitness, crossover_size FROM POLICY_DETAILS WHERE policy_id = ?', (policy_id,))
  fitness_lambda_data = cursor.fetchall()

  selected_policy_curve = plotly.graph_objects.Scatter(
    x=[d[0] for d in fitness_lambda_data],
    y=[d[1] for d in fitness_lambda_data],
    mode='lines+markers',
    name=f'Fitness-Lambda Policy {policy_id}',
    line=dict(color='blue', width=4)
  )

  data = [baseline_curve, selected_policy_curve]

  # Adding shaded area for variance if available
  if mean_initial_fitness is not None:
    upper_bound = plotly.graph_objects.Scatter(
      x=[mean_initial_fitness + std_dev_initial_fitness] * 2,
      y=[0, max([d[1] for d in fitness_lambda_data])],
      mode='lines',
      line=dict(width=0),
      showlegend=False
    )
    lower_bound = plotly.graph_objects.Scatter(
      x=[mean_initial_fitness - std_dev_initial_fitness] * 2,
      y=[0, max([d[1] for d in fitness_lambda_data])],
      mode='lines',
      fill='tonexty',
      fillcolor='rgba(0, 255, 0, 0.2)',
      line=dict(width=0),
      name='Variance Initial Fitness'
    )
    mean_line = plotly.graph_objects.Scatter(
      x=[mean_initial_fitness, mean_initial_fitness],
      y=[0, max([d[1] for d in fitness_lambda_data])],
      mode='lines',
      name=f'Mean Initial Fitness',
      line=dict(color='green', width=2, dash='dot')
    )
    data.extend([upper_bound, lower_bound, mean_line])

  layout = plotly.graph_objects.Layout(
    title=f'Fitness-Lambda Assignment for Policy {policy_id}',
    xaxis=dict(title='Fitness'),
    yaxis=dict(title='Lambda'),
    font=dict(family='Courier New, monospace', size=18, color='RebeccaPurple'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(245, 245, 245, 1)'
  )



  # ================= Print scalar info about specified policy: ====================================
  cursor.execute(f'SELECT policy_id, {xaxis_choice} FROM CONSTRUCTED_POLICIES WHERE policy_id = ?', (policy_id,))
  training_data = cursor.fetchone()
  assert len(training_data) > 0, f"Policy ID {policy_id} missing!"
  _, num_training_timesteps_or_num_training_fes = training_data
  cursor.execute(f'SELECT {yaxis_choice} FROM EVALUATION_EPISODES WHERE policy_id = ?', (policy_id,))
  evaluations = [e[0] for e in cursor.fetchall()]
  avg_evaluations = sum(evaluations) / len(evaluations) if evaluations else None
  std_dev = math.sqrt(sum((e - avg_evaluations) ** 2 for e in evaluations) / len(evaluations)) if evaluations else 0

  cursor.execute(f'SELECT {yaxis_choice} FROM EVALUATION_EPISODES WHERE policy_id = -1')
  baseline_evaluations = [e[0] for e in cursor.fetchall()]
  baseline_avg_length = sum(baseline_evaluations) / len(baseline_evaluations)
  baseline_variance = sum((e - baseline_avg_length) ** 2 for e in baseline_evaluations) / (len(baseline_evaluations) - 1) if len(baseline_evaluations) > 1 else 0
  baseline_std_dev = math.sqrt(baseline_variance)



  print("Policy ID:", policy_id)
  print("Y axis name:", yaxis_choice)
  print("Y axis:", "{:,}".format(num_training_timesteps_or_num_training_fes))
  print("Y axis (Mean):", avg_evaluations)
  print("Y axis (Mean - Stddev):", avg_evaluations - std_dev)
  print("Y axis (Mean + Stddev):", avg_evaluations + std_dev)
  print("Baseline y axis (Mean):", baseline_avg_length)
  print("Baseline y axis (Mean - Stddev):", baseline_avg_length - baseline_std_dev)
  print("Baseline y axis (Mean + Stddev):", baseline_avg_length + baseline_std_dev)
  # ================= PRINT END ====================================

  conn.close()
  fig = plotly.graph_objects.Figure(data=data, layout=layout)
  fig.show()

def print_matching(db_folder_path, filter_expression):
  configs = load_configs_from_folder(db_folder_path)
  matching_db_paths = filter_configs(configs, filter_expression)
  if not matching_db_paths:
    print("NO MATCH!")
    return
  for path in matching_db_paths:
    # print(path)
    pass
  return matching_db_paths

def display_config_as_dataframe(db_path):

  db_path = db_path.replace("\n", "")
  assert os.path.exists(db_path), f"There is no database at {db_path}!"

  try:
    with sqlite3.connect(db_path) as conn:
      df = pandas.read_sql_query("SELECT key, value FROM CONFIG", conn)
  except sqlite3.Error as e:
    print(f"SQLite error: {e}")
    return None

  def format_value(value):
    """Format numerical values with commas."""
    try:
      if isinstance(value, (int, float)):
        return "{:,}".format(value)
      if isinstance(value, str):
        if value.isdigit():
          return "{:,}".format(int(value))
        # Attempt to convert to float for strings like '123.45'
        try:
          return "{:,}".format(float(value))
        except ValueError:
          pass
    except (ValueError, TypeError):
      pass
    return value

  def unfold_dict(prefix, d, rows):
    """Recursive function to unfold nested dictionaries."""
    for key, value in d.items():
      new_key = f"{prefix}__{key}" if prefix else key
      if isinstance(value, dict):
        unfold_dict(new_key, value, rows)
      else:
        formatted_value = format_value(value)
        rows.append({'key': new_key, 'value': formatted_value})

  # Process each row and unfold nested dictionaries
  new_rows = []
  for _, row in df.iterrows():
    try:
      value = ast.literal_eval(row['value'])
      if isinstance(value, dict):
        unfold_dict(row['key'], value, new_rows)
      else:
        formatted_value = format_value(value)
        new_rows.append({'key': row['key'], 'value': formatted_value})
    except (ValueError, SyntaxError):
      # Format value if it's a number, keep as is otherwise
      formatted_value = format_value(row['value'])
      new_rows.append({'key': row['key'], 'value': formatted_value})

  # Create a new DataFrame from the processed rows and set 'key' as the index
  new_df = pandas.DataFrame(new_rows).set_index('key')

  return new_df

def diagrams(db_path_str_or_db_paths_list, xaxis_choice, yaxis_choice, policy_total_timesteps = None):
  if type(db_path_str_or_db_paths_list) == str:
    db_path_str = db_path_str_or_db_paths_list
    print(display_config_as_dataframe(db_path_str))
    policy_performance(db_path_str, xaxis_choice, yaxis_choice)
    if policy_total_timesteps is not None:
      generate_fitness_lambda_plot(db_path_str, policy_total_timesteps, xaxis_choice, yaxis_choice)
  if type(db_path_str_or_db_paths_list) == list:
    db_paths_list = db_path_str_or_db_paths_list
    for db_path in db_paths_list:
      print(display_config_as_dataframe(db_path))
      policy_performance(db_path, xaxis_choice, yaxis_choice)
      if policy_total_timesteps is not None:
        generate_fitness_lambda_plot(db_path, policy_total_timesteps, xaxis_choice, yaxis_choice)


import sqlite3
import numpy as numpy
import plotly.graph_objects as go
import os

def compute_area_between_baseline_and_policy(db_path):
    # Connect to the database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Fetch the baseline value for policy ID -1
        cursor.execute('''
            SELECT AVG(num_function_evaluations)
            FROM EVALUATION_EPISODES
            WHERE policy_id = -1
        ''')
        fetched = cursor.fetchone()
        baseline_value = fetched[0] if fetched else 0

        # Fetch average evaluations for each policy excluding ID -1 and calculate area
        cursor.execute('''
            SELECT policy_id, AVG(num_function_evaluations)
            FROM EVALUATION_EPISODES
            WHERE policy_id >= 0
            GROUP BY policy_id
            ORDER BY policy_id
        ''')
        eval_results = cursor.fetchall()

        # If no results, return area as zero
        if not eval_results:
            return 0

        # Extract evaluations and create timestep indices
        avg_evaluations = numpy.array([result[1] for result in eval_results])
        timestep_names = numpy.arange(len(avg_evaluations))
        baseline_array = numpy.full(avg_evaluations.shape, baseline_value)

        # Compute the difference and integrate to find the area
        difference = avg_evaluations - baseline_array
        area = numpy.trapz(difference, timestep_names)

        return area

def visualize_areas(area_values, db_paths):
  # Extract the basenames of the file paths without extension for labeling
  basenames = [os.path.splitext(os.path.basename(path))[0] for path in db_paths]

  # Create a bar chart using Plotly
  fig = plotly.graph_objects.Figure([plotly.graph_objects.Bar(x=basenames, y=area_values)])
  fig.update_layout(title='Area Between Baseline and Policy Evaluations Across Databases',
                    xaxis_title='Database',
                    yaxis_title='Computed Area',
                    showlegend=False)
  fig.show()

def create_multi_policy_performance_boxplot(paths, yaxis_choice="num_function_evaluations"):
  fig = plotly.graph_objects.Figure()

  for db_path in paths:
    with sqlite3.connect(db_path) as conn:
      cursor = conn.cursor()
      # Query for policies with ID >= 0
      cursor.execute(f'''
          SELECT policy_id, AVG({yaxis_choice}) as avg_eval
          FROM EVALUATION_EPISODES
          WHERE policy_id >= 0
          GROUP BY policy_id
      ''')
      results = cursor.fetchall()

      # Query for policy with ID -1
      cursor.execute(f'''
          SELECT policy_id, AVG({yaxis_choice}) as avg_eval
          FROM EVALUATION_EPISODES
          WHERE policy_id = -1
          GROUP BY policy_id
      ''')
      result_negative_one = cursor.fetchone()

    # Extract policy IDs and their corresponding average evaluations
    policy_ids = [result[0] for result in results]
    avg_evaluations = [result[1] for result in results]

    # Determine the highest policy ID and its average
    if policy_ids:  # Check if list is not empty
      max_policy_id = max(policy_ids)
      max_index = policy_ids.index(max_policy_id)
      max_avg = avg_evaluations[max_index]

      # Extract the average for policy ID -1
      avg_eval_negative_one = result_negative_one[1] if result_negative_one else None

      # Extract filename from the path for x-axis labeling
      filename = os.path.basename(db_path)

      # Add boxplot for each database
      fig.add_trace(plotly.graph_objects.Box(
          y=avg_evaluations,
          name=filename,  # Use filename as the x-value (category)
          boxpoints='all',
          jitter=0.5,
          pointpos=0,
          hoverinfo='y+name',
          hovertemplate=f'File: {filename}<br>Avg: {{y}}<br>Policy ID: {{text}}',
          text=policy_ids  # Pass policy IDs for hover template
      ))

      # Add a special marker for the highest policy ID
      fig.add_trace(plotly.graph_objects.Scatter(
          x=[filename],  # Match the x-value with the boxplot
          y=[max_avg],
          mode='markers',
          marker=dict(symbol='star', size=12, color='red'),
          name=f'Policy {max_policy_id} Max Avg in {filename}',
          hoverinfo='name+y',
          hovertemplate=f'File: {filename}<br>Max Avg: {{y}}<br>Policy ID: {max_policy_id}'
      ))

      # Add a special marker for policy ID -1
      if avg_eval_negative_one is not None:
          fig.add_trace(plotly.graph_objects.Scatter(
              x=[filename],
              y=[avg_eval_negative_one],
              mode='markers',
              marker=dict(symbol='triangle-up', size=12, color='yellow'),
              name=f'Policy -1 Avg in {filename}',
              hoverinfo='name+y',
              hovertemplate=f'File: {filename}<br>Avg: {{y}}<br>Policy ID: -1'
          ))

  # Customize layout
  fig.update_layout(
      title='Average Function Evaluations by Policy Across Databases',
      xaxis_title='Database Files',
      yaxis_title='Average Number of Function Evaluations',
      showlegend=True
  )

  fig.show()

def compute_hitting_times(paths, error_margin, yaxis_choice="num_function_evaluations"):
    hitting_times_list = []
    total_policies_list = []

    for db_path in paths:
        hitting_times = 0
        total_policies = 0

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'''
                SELECT policy_id, AVG({yaxis_choice}) as avg_eval
                FROM EVALUATION_EPISODES
                WHERE policy_id >= 0
                GROUP BY policy_id
            ''')
            results = cursor.fetchall()

            cursor.execute(f'''
                SELECT AVG({yaxis_choice}) as avg_eval,
                       GROUP_CONCAT({yaxis_choice}) as evaluations
                FROM EVALUATION_EPISODES
                WHERE policy_id = -1
            ''')
            result_negative_one = cursor.fetchone()

            if result_negative_one:
                baseline_avg = result_negative_one[0]
                baseline_evaluations = list(map(float, result_negative_one[1].split(',')))
                baseline_stddev = numpy.std(baseline_evaluations)

            total_policies = len(results)

            for result in results:
                policy_id = result[0]
                avg_evaluation = result[1]

                if result_negative_one and avg_evaluation < baseline_avg + error_margin * baseline_stddev:
                    hitting_times += 1

        hitting_times_list.append(hitting_times)
        total_policies_list.append(total_policies)

    return hitting_times_list, total_policies_list

def experiments_summary(experiment_paths):

  create_multi_policy_performance_boxplot(experiment_paths)

  error_margin = 0

  # Compute hitting times and total number of policies for all database paths
  hitting_times_list, total_policies_list = compute_hitting_times(experiment_paths, error_margin)

  # Calculate the difference between total policies and hitting policies count
  missing_policies_list = [total_policies - hitting_times for total_policies, hitting_times in zip(total_policies_list, hitting_times_list)]

  # Create bar chart using Plotly
  fig = plotly.graph_objects.Figure()

  # Add total policies bar (blue bar, above)
  fig.add_trace(plotly.graph_objects.Bar(x=[os.path.basename(path) for path in experiment_paths],
                       y=hitting_times_list,
                       name="Hitting Times"))

  # Add hitting times bar (red bar, below)
  fig.add_trace(plotly.graph_objects.Bar(x=[os.path.basename(path) for path in experiment_paths],
                       y=missing_policies_list,
                       name="Missing Policies"))

  fig.update_layout(title="Hitting Times vs Total Policies",
                    xaxis_title="Database Paths",
                    yaxis_title="Count",
                    barmode='stack',  # Stack bars vertically
                    bargap=0.1,       # Gap between bars of adjacent location coordinates
                    bargroupgap=0.1)  # Gap between bars of the same location coordinate
  fig.show()

  area_values = [compute_area_between_baseline_and_policy(db) for db in experiment_paths]
  visualize_areas(area_values, experiment_paths)
