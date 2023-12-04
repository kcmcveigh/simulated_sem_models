# %%
import pandas as pd
import numpy as np
from semopy import Model
import semopy as sem

# Function to simulate data based on the SEM models
def simulate_data(model_number, n=1000):
    np.random.seed(42)  # For reproducibility
    # Randomly generate the basic columns
    ideal_col = np.random.normal(0, 1, n)
    eval_col = np.random.normal(0, 1, n)
    pleasant_col = np.random.normal(0, 1, n)
    actual_col = np.random.normal(0, 1, n)  # Initialized but will be recalculated

    if model_number == 1:
        # {ideal_col} ~ {eval_col} + {pleasant_col}
        # {actual_col} ~ {ideal_col}
        ideal_col = 0.5 * eval_col + 0.5 * pleasant_col + np.random.normal(0, 0.5, n)
        actual_col = 0.7 * ideal_col + np.random.normal(0, 0.5, n)
    
    elif model_number == 2:
        # {actual_col} ~ {ideal_col} + {eval_col} + {pleasant_col}
        actual_col = 0.3 * ideal_col + 0.3 * eval_col + 0.3 * pleasant_col + np.random.normal(0, 0.5, n)

    elif model_number == 3:
        # {pleasant_col} ~ {ideal_col} + {pleasant_col}
        # {actual_col} ~ {pleasant_col}
        pleasant_col = 0.5 * ideal_col + 0.5 * pleasant_col + np.random.normal(0, 0.5, n)
        actual_col = 0.7 * pleasant_col + np.random.normal(0, 0.5, n)

    elif model_number == 4:
        # {eval_col} ~ {ideal_col} + {pleasant_col}
        # {actual_col} ~ {eval_col}
        eval_col = 0.5 * ideal_col + 0.5 * pleasant_col + np.random.normal(0, 0.5, n)
        actual_col = 0.7 * eval_col + np.random.normal(0, 0.5, n)

    # Create a DataFrame
    data = pd.DataFrame({
        'ideal_col': ideal_col,
        'eval_col': eval_col,
        'pleasant_col': pleasant_col,
        'actual_col': actual_col
    })
    return data

# Simulate data for each model
data_model1 = simulate_data(1)
data_model2 = simulate_data(2)
data_model3 = simulate_data(3)
data_model4 = simulate_data(4)

# Simulate data that does not conform to any model
data_non_conforming = pd.DataFrame({
    'ideal_col': np.random.normal(0, 1, 1000),
    'eval_col': np.random.normal(0, 1, 1000),
    'pleasant_col': np.random.normal(0, 1, 1000),
    'actual_col': np.random.normal(0, 1, 1000)
})

# %% Show the first few rows of the data for Model 1 as an example
for i, emo_df in enumerate([data_model1, data_model2, data_model3, data_model4, data_non_conforming]):
    model1_syntax = f"""
        ideal_col ~ eval_col + pleasant_col
        actual_col ~ ideal_col 
    """
    model1 = Model(model1_syntax)
    results1 = model1.fit(emo_df)
    model1_stats = sem.calc_stats(model1)

    model2_syntax = f"""
        actual_col ~ ideal_col + eval_col + pleasant_col
    """
    model2 = Model(model2_syntax)
    results2 = model2.fit(emo_df)
    model2_stats = sem.calc_stats(model2)

    model3_syntax = f"""
        pleasant_col ~ ideal_col + pleasant_col
        actual_col ~ pleasant_col 
    """
    model3 = Model(model3_syntax)
    results3 = model3.fit(emo_df)
    model3_stats = sem.calc_stats(model3)

    model4_syntax = f"""
        eval_col ~ ideal_col + pleasant_col
        actual_col ~ eval_col 
    """
    model4 = Model(model4_syntax)
    results4 = model4.fit(emo_df)
    model4_stats = sem.calc_stats(model4)

    df =pd.concat([model1_stats,model2_stats,model3_stats,model4_stats],axis=1)
    df.to_csv('simulated_model_comparison_stats_'+str(i)+'.csv')
# %%
