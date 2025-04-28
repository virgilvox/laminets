import matplotlib.pyplot as plt

def visualize_field(points):
    positions = [p.position.detach().cpu().numpy() for p in points]
    xs, ys = zip(*[(p[0], p[1]) for p in positions])

    plt.scatter(xs, ys, c='blue', alpha=0.5)
    plt.title('Field Point Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
