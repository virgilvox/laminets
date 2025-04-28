import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_field_evolution(field, steps=50, dt=0.01):
    fig, ax = plt.subplots()
    scat = ax.scatter([], [], c='blue', alpha=0.5)

    def init():
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        return scat,

    def update(frame):
        field.evolve(dt)
        positions = torch.stack([p.position for p in field.points]).detach().cpu().numpy()
        scat.set_offsets(positions[:, :2])  # only x and y
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, blit=True)
    plt.show()

# Usage inside training:

# After model evolves for a few steps:
# animate_field_evolution(model.last_field)
