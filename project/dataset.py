# %%bash
# cat > project/dataset.py <<'PY'
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_dataset(n=150, seed=42):
    # Seeds
    np.random.seed(seed)
    random.seed(seed)

    regions = ['North','South','East','West', None]
    categories = ['A','B','C','D', None]

    # Base data
    sales = np.random.normal(200, 50, n).round(2)
    profit = np.random.normal(40, 15, n).round(2)
    customers = np.random.randint(50, 150, n)
    satisfaction = np.random.normal(4.5, 0.3, n).round(1)

    # Outliers
    sales_outliers = [1000, 1200, 1500]
    profit_outliers = [200, -50, 0]
    customers_outliers = [500, 0, 100]
    satisfaction_outliers = [3.0, 5.0, 4.0]

    sales = np.concatenate([sales, sales_outliers])
    profit = np.concatenate([profit, profit_outliers])
    customers = np.concatenate([customers, customers_outliers])
    satisfaction = np.concatenate([satisfaction, satisfaction_outliers])

    # Categorical values
    region = [random.choice(regions) for _ in range(n)] + ['North','South','East']
    category = [random.choice(categories) for _ in range(n)] + ['A','B','C']

    # Dates in last 180 days
    date = [datetime.today() - timedelta(days=random.randint(0,180)) for _ in range(n+3)]

    # Build DataFrame
    df = pd.DataFrame({
        'date': date,
        'sales': sales,
        'profit': profit,
        'customers': customers,
        'region': region,
        'category': category,
        'satisfaction': satisfaction
    })

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    return df