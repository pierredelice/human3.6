from utils.read_params import read_params
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join

params = read_params()
file = "loss.csv"
file = join(
    params["results_dir"],
    file,
)
data = read_csv(file,
                index_col=0)
plt.subplots(figsize=(10, 4))
plt.plot(data["Validation"],
         label="Validation",
         color="#0D3B66")
plt.plot(data["Train"],
         label="Train",
         color="#F95738")
plt.ylim(0.035, 0.075)
plt.xticks(range(0, 5500, 500))
plt.xlabel("Iterations")
plt.ylabel("Loss function")
plt.grid(ls="--",
         color="#000000",
         alpha=0.6)
plt.tight_layout()
plt.savefig("test.png")
