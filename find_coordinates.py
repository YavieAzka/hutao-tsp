import matplotlib.pyplot as plt

img = plt.imread('D:\Kulyeah\Sem-2\Matematika Diskrit\Makalah\Sourcecode Matdis\marked_node.png')
coords = []

def onclick(event):
    x, y = int(event.xdata), int(event.ydata)
    coords.append((x, y))
    print(f"Clicked: ({x}, {y})")
    if len(coords) == 37:  # set this to the number of nodes you want
        plt.close()

plt.imshow(img)
plt.title("Click on the nodes (order matters)")
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Save coordinates
with open("node_coordinates.txt", "w") as f:
    for i, (x, y) in enumerate(coords):
        f.write(f"{i},{x},{y}\n")
