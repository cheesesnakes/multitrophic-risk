from PIL import Image
import os

# Function to make figures for publication


def make_figure(scenarios, plots, figure_name, rows=1, cols=3):
    os.makedirs("output/experiments/figures", exist_ok=True)
    dir = "output/experiments/plots"

    # Get the plots
    images = []
    for plot in plots:
        for scenario in scenarios:
            images.append(Image.open(f"{dir}/Scenario-{scenario}_{plot}.png"))

    # Create grid
    width = max([img.width for img in images])
    height = max([img.height for img in images])

    grid_image = Image.new("RGB", (cols * width, rows * height))

    for i, img in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(img, (x, y))

    # Save the grid image
    grid_image.save(f"output/experiments/figures/{figure_name}.png")
    print(f"{figure_name} created successfully.")

    return 0

def make_comparisons_figure():
    os.makedirs("output/experiments/figures", exist_ok=True)
    dir = "output/experiments/comparison"

    # Get the plots
    images = []
    
    images.append(Image.open(f"{dir}/state_All.png"))
    images.append(Image.open(f"{dir}/period_All.png"))

    # Create grid
    width = max([img.width for img in images])
    height = max([img.height for img in images])

    grid_image = Image.new("RGB", (width, 2 * height))

    for i, img in enumerate(images):
        x = 0
        y = i * height
        grid_image.paste(img, (x, y))

    # Save the grid image
    grid_image.save(f"output/experiments/figures/comparisons_figure.png")
    print("comparisons_figure created successfully.")

    return 0

def make_bifurcation_figure(configs):
    os.makedirs("output/experiments/figures", exist_ok=True)
    dir = "output/experiments/plots"

    # Get the plots

    tests = [1, 2, 3, 5, 6, 8, 9, 10]

    pops = ["prey", "predator", "apex"]

    for test in tests:
        images = []
    
        name = "Test-" + str(test)

        variable = configs[name]["variables"][0]

        for pop in pops:
            if os.path.exists(f"{dir}/{name}_bifurcation_{variable}_{pop}.png"):
                images.append(Image.open(f"{dir}/{name}_bifurcation_{variable}_{pop}.png"))
        
        # Create grid
        width = max([img.width for img in images])
        height = max([img.height for img in images])

        grid_image = Image.new("RGB", (width, len(images)*height))

        for i, img in enumerate(images):
            x = 0
            y = i * height
            grid_image.paste(img, (x, y))
        # Save the grid image
        grid_image.save(f"output/experiments/figures/{name}_bifurcation.png")

        print(f"Bifurcation figure for {name} created successfully.")
    
    return 0

if __name__ == "__main__":
    # Figure 3

    make_figure([0, 1, 6], ["phase_probability"], "figure3a")
    make_figure([0, 1, 6], ["timeseries"], "figure3b")

    # Figure 4
    make_figure(range(2, 8), ["phase_probability"], "figure4a", rows=2, cols=3)
    make_figure(range(2, 8), ["timeseries"], "figure4b", rows=2, cols=3)

    # Comparisons figure
    make_comparisons_figure()