import os, time, random
import pandas as pd
import numpy as np
from tqdm import tqdm

class BenchMark:
    file = None
    dataframe = None

    def __init__(self, name):
        self.name = name
        self.check_file()
        self.start_time = time.time()
        print("==== START BENCHMARK ====")

    def check_file(self):
        if not os.path.isfile("benchmarks.csv"):
            self.dataframe = pd.DataFrame(columns=["name", "time (s)"])
            self.dataframe.to_csv("benchmarks.csv", index=False)
        else:
            self.dataframe = pd.read_csv("benchmarks.csv")

    def stopBenchMark(self):
        duration = time.time() - self.start_time
        self.dataframe = self.dataframe.append({"name": self.name, "time (s)": duration}, ignore_index=True)
        print("==== STOP BENCHMARK ====")
        self.dataframe.to_csv("benchmarks.csv", index=False)
