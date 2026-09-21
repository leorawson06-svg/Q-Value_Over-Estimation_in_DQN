import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############ NOTE ##############

"This script specifically is basically written by ChatGPT, "
"I do not claim credit for this script"
"I Included in case simple plots needed to be replicated "




def plot_quantity(quantity, df):

    # --------------------------------------------------
    # Calculate mean/std across seeds
    # --------------------------------------------------

    summary = (
        df
        .groupby([
            "algorithm",
            "target_update_interval",
            "training_step",
        ])[quantity]
        .agg(
            mean="mean",
            std="std",
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------
    # Line style represents algorithm
    # --------------------------------------------------

    line_styles = {
        "DQN": "-",
        "DDQN": "--",
    }

    # --------------------------------------------------
    # Assign ONE colour to each K
    #
    # We let matplotlib generate the colours automatically.
    # DQN/DDQN with the same K then reuse the same colour.
    # --------------------------------------------------

    frequencies = sorted(
        summary["target_update_interval"].unique()
    )

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    frequency_colors = {
        frequency: default_colors[i % len(default_colors)]
        for i, frequency in enumerate(frequencies)
    }

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    for frequency in frequencies:

        for algorithm in ["DQN", "DDQN"]:

            data = summary[
                (summary["algorithm"] == algorithm)
                &
                (summary["target_update_interval"] == frequency)
            ].sort_values("training_step")

            if data.empty:
                continue

            x = data["training_step"]
            mean = data["mean"]
            std = data["std"]

            color = frequency_colors[frequency]

            ax.plot(
                x,
                mean,
                color=color,
                linestyle=line_styles[algorithm],
                linewidth=2,
                label=f"K = {frequency}, {algorithm}",
            )

            # +/- one standard deviation
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=0.08,
            )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_xlabel("Training step")

    ax.set_ylabel(
        quantity.replace("_", " ")
    )

    ax.set_title(
        quantity.replace("_", " "),
        fontsize=15,
        fontweight="bold",
    )

    ax.grid(alpha=0.2)

    ax.legend(
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    # plt.savefig(
    #  
    #     f"Cartpole_Oversestimate/results/figures/{quantity}.png",
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    plt.show()


POL_data_path = r"\results\policy_metrics_working.csv"
Q_data_path = r"\results\q_metrics_working.csv"
POL_2_data_path = r"\results\policy_metrics.csv"
# plot_quantity("Avg_Reward",
#     POL_data_path
    
# )


# plot_quantity("Avg_Abs_Diff",
#     POL_data_path
    
# )

df_1 = pd.read_csv(POL_2_data_path) #+ pd.read_csv(POL_2_data_path)
df_2 = pd.read_csv(POL_data_path)
df_3 = pd.read_csv(r"results\policy_metrics_2.csv")


df = pd.concat([df_1, df_2,df_3], ignore_index=True)

# Save
df.to_csv("combined.csv", index=False)

df = df.rename(columns={
    "Mean_Violtaing_Excess": "Mean_Violating_Excess",
    "Mean_Under_Violtaion_Rate": "Mean_Under_Violation_Rate",
    "Avg_Abs_Diff":"Intial_Q_-_G_(Excess)"
})

cols = ['Avg_Reward', 'Intial_Q_-_G_(Excess)']
cols = list(df.columns)
del cols[:5]

# plot_quantity('Mean_Excess',df)

for col in cols:
    plot_quantity(col,
        df
    )







