from tqdm import tqdm
import random
import time


BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} {rate_fmt}{postfix}'
with tqdm(desc='Epoch', total=10, bar_format=BAR_FORMAT, leave=False) as outer_bar:
    for j in range(10):
        with tqdm(desc="Batch", total=100, bar_format=BAR_FORMAT, leave=False) as inner_bar:
            for i in range(100):
                time.sleep(0.1)
                inner_bar.update(1)
                inner_bar.set_postfix({f"Metric {i+1}": random.choice(["↑", "↓"]) for i in range(4)}, refresh=True)
            outer_bar.update(1)
