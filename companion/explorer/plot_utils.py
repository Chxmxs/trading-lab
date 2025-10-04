import matplotlib.pyplot as plt

def plot_equity(equity, title="Equity Curve", filename="equity.png"):
    """
    Generate and save a simple equity curve plot.
    Parameters:
    - equity (pd.Series): Equity curve indexed by date.
    - title (str): Title for the plot.
    - filename (str): Name of the output PNG file.
    """
    plt.figure()
    equity.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.savefig(filename)
    plt.close()
